#!/usr/bin/env python
# coding: utf-8

# In[153]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace 0 with your GPU ID


# In[154]:


import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import gymnax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from wrappers import (
    GymnaxWrapper,
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    #env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train




# In[155]:


print(jax.devices())


# In[156]:


config = { # not very good config
    "LR": 3e-4,
    "NUM_ENVS": 8192, #4096,#2048,
    "NUM_STEPS": 20, #10,
    "TOTAL_TIMESTEPS": 409600 * 200,#5e7,
    "UPDATE_EPOCHS": 10, #4,
    "NUM_MINIBATCHES": 64, #32,
    "GAMMA": 0.98, #0.99,
    "GAE_LAMBDA": 0.93,#0.95,
    "CLIP_EPS": 0.15,#0.2,
    "ENT_COEF": 0.01,#0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",#"tanh",
    "ENV_NAME": "TwoStageOTA-custom",
    "ANNEAL_LR": True,#False,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
}
config = {
    "LR": 3e-4,
    "NUM_ENVS": 4096,#2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 409600 * 60,#5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh", #"relu",#"tanh",
    "ENV_NAME": "TwoStageOTA-custom",
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
}
config = { # best config so far found from optuna_hpsearch_ppo.py
    'LR': 0.00044574140508460696, 
    "NUM_ENVS": 4096,#2048,
    'NUM_STEPS': 15, 
    "TOTAL_TIMESTEPS": 409600 * 60,#5e7,
    'UPDATE_EPOCHS': 4, 
    'NUM_MINIBATCHES': 64, 
    'GAMMA': 0.9719202527669172, 
    'GAE_LAMBDA': 0.9361494624548591, 
    'CLIP_EPS': 0.2874919405095487, 
    'ENT_COEF': 2.1685069159216622e-05, 
    'VF_COEF': 0.5822412408290565, 
    'MAX_GRAD_NORM': 0.42710241041977387, 
    'ACTIVATION': 'tanh',
    "ENV_NAME": "TwoStageOTA-custom",
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
    }
print(jax.devices())

rng = jax.random.PRNGKey(30)
train_jit = jax.jit(make_train(config))
#out = train_jit(rng)


# In[ ]:


import time
import matplotlib.pyplot as plt
rng = jax.random.PRNGKey(42)
t0 = time.time()
out = jax.block_until_ready(train_jit(rng))
print(f"time: {time.time() - t0:.2f} s")
plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Return")
plt.show()
plt.savefig('foo.png')

# Also plot y log axis graph
avg_returns = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
# Remove initial zeros
no_zeros = avg_returns[avg_returns < 0]
plt.plot(-no_zeros)
plt.yscale('log')  # Setting y-axis to logarithmic scale
plt.title("Log absolute y scale (initial 0s removed)")
plt.xlabel("Update Step")
plt.ylabel("Return (log)")
plt.savefig("foo_log.png")

# Plot best sample
plt.figure(figsize=(20, 12))

return_rewards = out["metrics"]["returned_episode_returns"].reshape(-1)
non_zero_samples = return_rewards[return_rewards < 0]
plt.plot(non_zero_samples)
plt.yscale('linear')  # Revert y-axis to lienar scale
plt.xlabel("Update Samples")
plt.ylabel("Return")
plt.show()
plt.savefig('foo_samples.png')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Data preparation
return_rewards = out["metrics"]["returned_episode_returns"].reshape(-1)
non_zero_samples = return_rewards[return_rewards < 0]

# Create a larger plot
plt.figure(figsize=(10, 6))

# Plot the full data
plt.plot(non_zero_samples, label='Return', color='blue', linewidth=2)
plt.xlabel("Update Samples", fontsize=14)
plt.ylabel("Return", fontsize=14)
plt.title("Return Over Update Samples", fontsize=16)
plt.grid(True)

# Create an inset of the larger plot
# Arguments are [x-coordinate, y-coordinate, width, height] relative to the parent plot
ax_inset = inset_axes(plt.gca(), width='40%', height='30%', loc='lower right')
ax_inset.plot(non_zero_samples, color='blue', linewidth=2)
ax_inset.set_xlim(0, len(non_zero_samples))  # Set x-limits to show the same x-range as the main plot
ax_inset.set_ylim(-5, 0)  # Set y-limits to focus on the 0~-5 section
ax_inset.set_title('Focus on 0~-5', fontsize=10)  # Add title to inset
ax_inset.grid(True)  # Add grid to inset for better readability

plt.tight_layout()  # Adjust subplot parameters for better layout
plt.savefig('foo_samples_with_focus.png')
plt.show()
# In[ ]:


out_runner_state = (out['runner_state']) # train_state, env_state, last_obs, rng
out_runner_env = ((out['runner_state'][1]))


# In[ ]:


print(out_runner_env.env_state.env_state.env_state)


# In[ ]:


a = out_runner_env.env_state.env_state.env_state.x11
plt.hist(a)


# In[ ]:


print(out_runner_env.env_state.env_state.env_state.x0)


# In[ ]:


out_runner_env.env_state


# In[ ]:


len(out_runner_state)


# In[ ]:


type(out_runner_state[0])


# In[ ]:


out_runner_state[2].shape # env_state


# In[ ]:


out_runner_state[2][0]


# In[ ]:


return_rewards_mean = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)


# In[ ]:


return_rewards_mean.shape


# In[ ]:


print('best batch reward',(return_rewards_mean[return_rewards_mean < 0]).max()) # best batch reward


# In[ ]:


out["metrics"]["returned_episode_returns"].shape


# In[ ]:


return_rewards = out["metrics"]["returned_episode_returns"].reshape(-1)


# In[ ]:


return_rewards.shape
print('return_rewards.shape', return_rewards.shape)


# In[ ]:


return_rewards


# In[ ]:


print('absolute best reward per sample', (return_rewards[return_rewards < 0]).max()) # absolute best reward per sample


# In[ ]:


# Testing to see if FoM implementation is correct
# Also check how much FoM diverges when using less accurate output result from mlp model as opposed to golden SPICE sim result

from typing import Tuple, Optional
import chex
# Measure array (reference value from simulation)
Measure = jnp.array([
    1.738700e+08,  # ugf
    1.342800e+02,  # pm
    6.144200e-08,  # dc_gain
    9.234700e+01,  # cmrr
    6.605900e+01,  # psrr
    1.114900e+02,  # os
    1.798600e+00,  # upper_trig
    1.779000e-02,  # lower_trig
    5.159700e-04   # pwr
])
# Model output array (output from estimation model)
mlp_output = jnp.array([ 1.72730023e+08, -1.30592169e+01,  3.13270706e-08,  1.03272513e+02,
  6.93153630e+01,  1.19041006e+02,  1.76345655e+00,  8.49051201e-02,
  5.16918122e-04])

weights = jnp.array([1, 0, 1, 1, 1, 1, 1, 1, 1])
class EnvParams:
    x0_bounds: Tuple[float, float] = (100e-15, 10000e-15)
    x1_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x2_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x3_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x4_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x5_bounds: Tuple[float, float] = (180e-9, 2e-6)
    x6_bounds: Tuple[float, float] = (100e-15, 2000e-15)
    x7_bounds: Tuple[float, float] = (1, 20)
    x8_bounds: Tuple[float, float] = (1, 20)
    x9_bounds: Tuple[float, float] = (1, 20)
    x10_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x11_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x12_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x13_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x14_bounds: Tuple[float, float] = (220e-9, 150e-6)
    x15_bounds: Tuple[float, float] = (1e2, 1e5)
    out0_constraints: Tuple[float, int] = (30e6, 0)
    out1_constraints: Tuple[float, int] = (60, 0)
    out2_constraints: Tuple[float, int] = (100e-9, 1)
    out3_constraints: Tuple[float, int] = (80, 0)
    out4_constraints: Tuple[float, int] = (60, 0)
    out5_constraints: Tuple[float, int] = (80, 0)
    out6_constraints: Tuple[float, int] = (1.5, 0)
    out7_constraints: Tuple[float, int] = (30e-3, 1)
    out0_denormalize: Tuple[float, float] = (0.0, 645470000.0) # min, max, were mapped to [-1,1]
    out1_denormalize: Tuple[float, float] = (-180.0, 179.99)
    out2_denormalize: Tuple[float, float] = (1.2585e-11, 1e-06)
    out3_denormalize: Tuple[float, float] = (-68.19, 210.7)
    out4_denormalize: Tuple[float, float] = (-181.83, 99.497)
    out5_denormalize: Tuple[float, float] = (-4.0129, 150.14)
    out6_denormalize: Tuple[float, float] = (-1.3605, 1.8004)
    out7_denormalize: Tuple[float, float] = (8.661e-06, 2.882)
    out8_denormalize: Tuple[float, float] = (9.6713e-05, 0.0047157)
    num_states: int = 16
    num_spects: int = 9
    max_steps_in_episode: int = 20

def compute_reward(model_output: chex.Array, params: EnvParams, weights) -> float:
    # Uses FoM metric for reward
    # Denormalize the model output to compare against constraints
    # From [-1, 1] => [min, max]
    denormalize_params_min = jnp.array([
        params.out0_denormalize[0], params.out1_denormalize[0], params.out2_denormalize[0],
        params.out3_denormalize[0], params.out4_denormalize[0], params.out5_denormalize[0],
        params.out6_denormalize[0], params.out7_denormalize[0], params.out8_denormalize[0]
    ])
    denormalize_params_max = jnp.array([
        params.out0_denormalize[1], params.out1_denormalize[1], params.out2_denormalize[1],
        params.out3_denormalize[1], params.out4_denormalize[1], params.out5_denormalize[1],
        params.out6_denormalize[1], params.out7_denormalize[1], params.out8_denormalize[1]
    ])

    constraints = jnp.array([
        params.out0_constraints, params.out1_constraints, params.out2_constraints,
        params.out3_constraints, params.out4_constraints, params.out5_constraints,
        params.out6_constraints, params.out7_constraints
    ])

    model_output = ((2 * (model_output - denormalize_params_min)) / (denormalize_params_max - denormalize_params_min)) - 1
    out = denormalize_params_min + (((model_output + 1.0) * (denormalize_params_max - denormalize_params_min)) / 2.0)
    #out = model_output * jnp.sqrt(denormalize_params[:, 1]) + denormalize_params[:, 0]

    # Calculate scaled differences from constraints
    scaled_diffs = jnp.where(
        constraints[:, 1] == 1,
        jnp.clip((out[:-1] - constraints[:, 0]) / constraints[:, 0], 0, 1),
        jnp.clip((constraints[:, 0] - out[:-1]) / constraints[:, 0], 0, 1)
    )
    # Append out[-1] to scaled_diffs
    scaled_diffs = jnp.append(scaled_diffs, out[-1])

    # Apply weights and sum
    weighted_diffs = weights * scaled_diffs
    FoM = jnp.sum(weighted_diffs)

    # Convert FoM to a reward
    reward = -1 * FoM

    return reward

reference_reward = compute_reward(Measure, EnvParams(), weights)


# In[ ]:


# Reference FoM implementation
import numpy as np

# Measure array
Measure = np.array([
    1.738700e+08,  # ugf
    1.342800e+02,  # pm
    6.144200e-08,  # dc_gain
    9.234700e+01,  # cmrr
    6.605900e+01,  # psrr
    1.114900e+02,  # os
    1.798600e+00,  # upper_trig
    1.779000e-02,  # lower_trig
    5.159700e-04   # pwr
])

# Weights for each measurement
weights = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1])

# Constraints
constraints = np.array([
    [30e6, 0],      # ugf
    [60, 0],        # pm
    [100e-9, 1],    # dc_gain
    [80, 0],        # cmrr
    [60, 0],        # psrr
    [80, 0],        # os
    [1.5, 0],       # upper_trig
    [30e-3, 1]      # lower_trig
])

# Calculate FoM
accum = 0
len_constraints = len(constraints)
for i in range(len_constraints):
    if constraints[i][1] == 1:
        accum += weights[i] * min(1.0, max(0.0, (Measure[i] - constraints[i][0]) / constraints[i][0]))
    else:
        accum += weights[i] * min(1.0, max(0.0, (constraints[i][0] - Measure[i]) / constraints[i][0]))

FoM_out = accum + Measure[len_constraints]  # Adding the pwr value
FoM_out


# In[ ]:


reference_reward


# In[ ]:


mlp_model_reward = compute_reward(mlp_output, EnvParams(), weights)


# In[ ]:


mlp_model_reward


# In[ ]:





# In[ ]:





# In[ ]:




