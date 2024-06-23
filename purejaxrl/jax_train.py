import logging
import os
import datetime

#import isaacgym

import hydra
from hydra.utils import to_absolute_path
#from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
#import gym
import sys
import shutil
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace 0 with your GPU ID
print('Test: os.environ["CUDA_VISIBLE_DEVICES"] = "0" executed')

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

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
#    GymnaxWrapper, # commented out as it is not used
    LogWrapper,
#    BraxGymnaxWrapper, # commented out as it is not used
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

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict



@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    print('Debug: launch_rlg_hydra executed')
    print(cfg)
    #time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)




    # sets seed. if seed is -1 will pick a random one
    rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.seed += rank
    cfg.train.params.config.multi_gpu = cfg.multi_gpu




    # Save the environment code!
    try:
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        gymnax_path = os.path.dirname(gymnax.__file__)
        output_file = os.path.join(gymnax_path, "environments", "custom", f"{cfg.task.env.env_name.lower()}.py")
        shutil.copy(output_file, f"env.py")
    except:
        import re
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"
        gymnax_path = os.path.dirname(gymnax.__file__)
        output_file = os.path.join(gymnax_path, "environments", "custom", f"{camel_to_snake(cfg.task.name)}.py")
        shutil.copy(output_file, f"env.py")



    if cfg.wandb_activate and rank ==0 :

        import wandb

        # initialize wandb only once per horovod run (or always for non-horovod runs)

    # dump config dict
    exp_date = cfg.train.params.config.name + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_dir = os.path.join('runs', exp_date)
    print("Network Directory:", Path.cwd() / experiment_dir / "nn")
    print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries")

    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


    # convert CLI arguments into dictionary
    # create runner and set the settings
    jax_train_config = { # best config so far found from optuna_hpsearch_ppo.py
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
    "ENV_NAME": "TwoStageOTAGPT-custom",
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": True,
    "DEBUG": False, # True # Changed to False because setting this to True pollutes the env_iter*_response*.txt with too much information
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(jax_train_config))
    import time
    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"time: {time.time() - t0:.2f} s")
    out_runner_env = ((out['runner_state'][1]))
    # select the best performing sample from final batch
    best_reward = np.max(out_runner_env.env_state.env_state.returned_episode_returns)
    print('best reward', best_reward)
    best_index = np.argmax(out_runner_env.env_state.env_state.returned_episode_returns)
    print('best index', best_index)
    # 99th percentile or top 1 % value
    top_1_val = np.percentile(out_runner_env.env_state.env_state.returned_episode_returns, 10)
    top_1_val_index = (np.where(out_runner_env.env_state.env_state.returned_episode_returns >= top_1_val)[0])[len((np.where(out_runner_env.env_state.env_state.returned_episode_returns >= top_1_val)[0])) // 2]
    print('top 1 % value', top_1_val)
    print('top 1 % index', top_1_val_index)
    # Iterate through the attribute names 'x0' to 'x15'
    best_sample_gate_config = []
    best_sample_gate_config_dict = {}
    best_sample_gate_config_dict_denorm = {}
    # using best_index seems to have it get stuck in a local minimum
    best_index = top_1_val_index
    for i in range(16):
        # Construct the attribute name
        attr_name = f"x{i}"

        # Use getattr to fetch the attribute from the env_state.env_state object
        value = getattr(out_runner_env.env_state.env_state.env_state, attr_name)[best_index]

        # Print the attribute in the desired format
        #print(f'{attr_name}: {value}')
        best_sample_gate_config_dict[attr_name] = value
        best_sample_gate_config.append(value)
    env, env_params = gymnax.make(jax_train_config["ENV_NAME"])
    denorm_states = env.deNorm_action(best_sample_gate_config, env_params)

    for i in range(16):
        print(f'x{i}: {denorm_states[i]}')
        best_sample_gate_config_dict_denorm[f'x{i}'] = denorm_states[i]

    statistics = {
        'best_reward': best_reward,
        'best_sample_gate_config_dict': best_sample_gate_config_dict,
        'best_sample_gate_config_dict_denorm': best_sample_gate_config_dict_denorm,
    }


    if cfg.wandb_activate and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    print('Debug: jax_train.py executed')
    launch_rlg_hydra()
    print('Finished running jax_train.py')



