import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from flax import linen as nn
import numpy as np


@struct.dataclass
class EnvState:
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float
    x9: float
    x10: float
    x11: float
    x12: float
    x13: float
    x14: float
    x15: float
    time: int


@struct.dataclass
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
    out0_constraints: Tuple[int, int] = (30e6, 0)
    out1_constraints: Tuple[float, float] = (60, 0)
    out2_constraints: Tuple[float, float] = (100e-9, 1)
    out3_constraints: Tuple[float, float] = (80, 0)
    out4_constraints: Tuple[float, float] = (60, 0)
    out5_constraints: Tuple[float, float] = (80, 0)
    out6_constraints: Tuple[float, float] = (1.5, 0)
    out7_constraints: Tuple[float, float] = (30e-3, 1)
    out0_denormalize: Tuple[float, float] = (5.17507048e+07, 8.35818696e+14) # mean, variance
    out1_denormalize: Tuple[float, float] = (1.65893357e+00, 2.68230064e+03)
    out2_denormalize: Tuple[float, float] = (5.11355348e-07, 1.50227783e-13)
    out3_denormalize: Tuple[float, float] = (1.22396347e+02, 2.39095780e+02)
    out4_denormalize: Tuple[float, float] = (8.06248731e+01, 2.12710884e+02)  
    out5_denormalize: Tuple[float, float] = (1.19612336e+02, 2.31070948e+02)
    out6_denormalize: Tuple[float, float] = (1.16087911e+00, 4.04046009e-01)
    out7_denormalize: Tuple[float, float] = (1.37376575e-01, 1.35465665e-02)
    out8_denormalize: Tuple[float, float] = (1.16796022e-03, 1.47283496e-07)
    num_states: int = 16
    num_spects: int = 9
    max_steps_in_episode: int = 20


class TwoStageOTA(environment.Environment):
    """
    JAX Compatible version of TwoStageOTA
    """
    class FlaxMLP(nn.Module):
        @nn.compact    
        def __call__(self, x):
              x = nn.Dense(512, name = 'Dense_0')(x)
              x = nn.silu(x)
              x = nn.Dense(512, name = 'Dense_1')(x)
              x = nn.silu(x)
              x = nn.Dense(256, name = 'Dense_2')(x)
              x = nn.silu(x)
              x = nn.Dense(128, name = 'Dense_3')(x)
              x = nn.silu(x)
              x = nn.Dense(9, name = 'Dense_4')(x)
              return x      
    def __init__(self):
        super().__init__()
        self.model = TwoStageOTA.FlaxMLP()
        key = jax.random.PRNGKey(0)
        params = self.model.init(key, jnp.ones((1,16)))['params']
        self.model_params = self.load_model_params()
        print(self.model_params)
        for layer_name, layer_params in self.model_params.items():
               if layer_name in params:
                     params[layer_name].update(layer_params)

    def load_model_params(self):
        new_params = {
            'Dense_0': {'kernel': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.0.weight.npy').T, 'bias': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.0.bias.npy')},
            'Dense_1': {'kernel': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.2.weight.npy').T, 'bias': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.2.bias.npy')},
            'Dense_2': {'kernel': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.4.weight.npy').T, 'bias': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.4.bias.npy')},
            'Dense_3': {'kernel': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.6.weight.npy').T, 'bias': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.6.bias.npy')},
            'Dense_4': {'kernel': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.8.weight.npy').T, 'bias': np.load('/home/sjpark/1_Reserch/3_GPU/gymnax_Analog_RL/gymnax/environments/custom/layers.8.bias.npy')}
        }
        return new_params


    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def deNorm_action(
        current_state: EnvState,
        params: EnvParams,
    ) -> EnvState:
        # Denoramlize the action based on the bounds within param
        x0 = (current_state.x0 + 1) * ((params.x0_bounds[1] - params.x0_bounds[0]) / 2) + params.x0_bounds[0]
        x1 = (current_state.x1 + 1) * ((params.x1_bounds[1] - params.x1_bounds[0]) / 2) + params.x1_bounds[0]
        x2 = (current_state.x2 + 1) * ((params.x2_bounds[1] - params.x2_bounds[0]) / 2) + params.x2_bounds[0]
        x3 = (current_state.x3 + 1) * ((params.x3_bounds[1] - params.x3_bounds[0]) / 2) + params.x3_bounds[0]
        x4 = (current_state.x4 + 1) * ((params.x4_bounds[1] - params.x4_bounds[0]) / 2) + params.x4_bounds[0]
        x5 = (current_state.x5 + 1) * ((params.x5_bounds[1] - params.x5_bounds[0]) / 2) + params.x5_bounds[0]
        x6 = (current_state.x6 + 1) * ((params.x6_bounds[1] - params.x6_bounds[0]) / 2) + params.x6_bounds[0]
        x7 = (current_state.x7 + 1) * ((params.x7_bounds[1] - params.x7_bounds[0]) / 2) + params.x7_bounds[0]
        x8 = (current_state.x8 + 1) * ((params.x8_bounds[1] - params.x8_bounds[0]) / 2) + params.x8_bounds[0]
        x9 = (current_state.x9 + 1) * ((params.x9_bounds[1] - params.x9_bounds[0]) / 2) + params.x9_bounds[0]
        x10 = (current_state.x10 + 1) * ((params.x10_bounds[1] - params.x10_bounds[0]) / 2) + params.x10_bounds[0]
        x11 = (current_state.x11 + 1) * ((params.x11_bounds[1] - params.x11_bounds[0]) / 2) + params.x11_bounds[0]
        x12 = (current_state.x12 + 1) * ((params.x12_bounds[1] - params.x12_bounds[0]) / 2) + params.x12_bounds[0]
        x13 = (current_state.x13 + 1) * ((params.x13_bounds[1] - params.x13_bounds[0]) / 2) + params.x13_bounds[0]
        x14 = (current_state.x14 + 1) * ((params.x14_bounds[1] - params.x14_bounds[0]) / 2) + params.x14_bounds[0]
        x15 = (current_state.x15 + 1) * ((params.x15_bounds[1] - params.x15_bounds[0]) / 2) + params.x15_bounds[0]
        return EnvState(
            x0=x0,
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x5=x5,
            x6=x6,
            x7=x7,
            x8=x8,
            x9=x9,
            x10=x10,
            x11=x11,
            x12=x12,
            x13=x13,
            x14=x14,
            x15=x15,
            time=current_state.time,
        )

    def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:
        # Uses FoM metric for reward
        # Denormalize the model output to compare against constraints
        out0 = model_output[0] * jnp.sqrt(params.out0_denormalize[1]) + params.out0_denormalize[0]
        out1 = model_output[1] * jnp.sqrt(params.out1_denormalize[1]) + params.out1_denormalize[0]
        out2 = model_output[2] * jnp.sqrt(params.out2_denormalize[1]) + params.out2_denormalize[0]
        out3 = model_output[3] * jnp.sqrt(params.out3_denormalize[1]) + params.out3_denormalize[0]
        out4 = model_output[4] * jnp.sqrt(params.out4_denormalize[1]) + params.out4_denormalize[0]
        out5 = model_output[5] * jnp.sqrt(params.out5_denormalize[1]) + params.out5_denormalize[0]
        out6 = model_output[6] * jnp.sqrt(params.out6_denormalize[1]) + params.out6_denormalize[0]
        out7 = model_output[7] * jnp.sqrt(params.out7_denormalize[1]) + params.out7_denormalize[0]
        out8 = model_output[8] * jnp.sqrt(params.out8_denormalize[1]) + params.out8_denormalize[0]
        accum = 0.0
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out0_constraints[0] - out0) / params.out0_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out1_constraints[0] - out1) / params.out1_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (out2 - params.out2_constraints[0]) / params.out2_constraints[0])) # second value of constraint is 1
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out3_constraints[0] - out3) / params.out3_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out4_constraints[0] - out4) / params.out4_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out5_constraints[0] - out5) / params.out5_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (params.out6_constraints[0] - out6) / params.out6_constraints[0]))
        accum += 1 * jnp.min(1.0, jnp.max(0.0, (out7 - params.out7_constraints[0]) / params.out7_constraints[0])) # second value of constraint is 1
        accum += 1 * out8 # no constraints on this output as this is power and smaller is always better

        FoM = accum
        reward = -1 * FoM # We want to minimize FoM, so we negate it to get reward (as the RL agent is trying to maximize reward)
        return reward



    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        # Denormalize the action
        #action = self.deNorm_action(action, params)
        jax_state = self.get_obs(state)

        output = self.model.apply(self.model_params, jax_state)
        reward = self.compute_reward(output, params)

        # Update state
        next_state = EnvState(
            x0=action[0],
            x1=action[1],
            x2=action[2],
            x3=action[3],
            x4=action[4],
            x5=action[5],
            x6=action[6],
            x7=action[7],
            x8=action[8],
            x9=action[9],
            x10=action[10],
            x11=action[11],
            x12=action[12],
            x13=action[13],
            x14=action[14],
            x15=action[15],
            time=state.time + 1,
        )
        done = self.is_terminal(next_state, params)

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            reward,
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        # All state initialized to 0.0
        init_state = EnvState(
            x0=0.0, x1=0.0, x2=0.0, x3=0.0, x4=0.0, x5=0.0, x6=0.0, x7=0.0, 
            x8=0.0, x9=0.0, x10=0.0, x11=0.0, x12=0.0, x13=0.0, x14=0.0, x15=0.0, 
            time=0)
        return self.get_obs(init_state), init_state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x0, state.x1, state.x2, state.x3, state.x4, state.x5, state.x6, state.x7, state.x8, state.x9, state.x10, state.x11, state.x12, state.x13, state.x14, state.x15]).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoStageOTA-custom"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.default_params.num_states

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(params.num_states,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Box(low=-1.0, high=1.0, shape=(params.num_states,), dtype=jnp.float32)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict(
        {
            "x0": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x1": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x2": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x3": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x4": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x5": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x6": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x7": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x8": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x9": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x10": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x11": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x12": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x13": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x14": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "x15": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32),
            "time": spaces.Discrete(self.default_params.max_steps_in_episode),
	}
	)
