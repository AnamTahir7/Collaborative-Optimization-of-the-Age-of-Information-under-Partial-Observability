from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import numpy as np


def make_multi_agent(env_name_or_creator):
    """Convenience wrapper for any single-agent env to be converted into MA.
    Agent IDs are int numbers starting from 0 (first agent).
    Args:
        env_name_or_creator (Union[str, Callable[]]: String specifier or
            env_maker function.
    Returns:
        Type[MultiAgentEnv]: New MultiAgentEnv class to be used as env.
            The constructor takes a config dict with `num_agents` key
            (default=1). The reset of the config dict will be passed on to the
            underlying single-agent env's constructor.
    Examples:
         # >>> # By gym string:
         # >>> ma_cartpole_cls = make_multi_agent("CartPole-v0")
         # >>> # Create a 2 agent multi-agent cartpole.
         # >>> ma_cartpole = ma_cartpole_cls({"num_agents": 2})
         # >>> obs = ma_cartpole.reset()
         # >>> print(obs)
         # ... {0: [...], 1: [...]}
         # >>> # By env-maker callable:
         # >>> ma_stateless_cartpole_cls = make_multi_agent(
         # ...    lambda config: StatelessCartPole(config))
         # >>> # Create a 2 agent multi-agent stateless cartpole.
         # >>> ma_stateless_cartpole = ma_stateless_cartpole_cls(
         # ...    {"num_agents": 2})
    """

    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            self.env = env_name_or_creator(config)
            obs = self.env.observation_space
            self.observation_space = {i: obs for i in range(self.env.number_of_agents)}
            self.action_space = self.env.action_space
            self.cnt = 0

        def reset(self):
            obs_state = self.env.reset()
            if self.env.config == 'na':
                ma_obs = {i: [obs_state[i]] for i in range(self.env.number_of_agents)}
            else:
                agent_aoi_states = obs_state[0]
                agent_ch_states = obs_state[1]
                ma_obs = {i: (np.append(agent_aoi_states[i], agent_ch_states[i])) for i in range(self.env.number_of_agents)}
            return ma_obs

        def step(self, action_dict):
            ma_obs, rew, dones, info = {}, {}, {}, {}
            action_list = []
            for i, action in action_dict.items():
                action_list.append(action)
            obs_state, avg_reward, done, _ = self.env.step(action_list)
            agent_aoi_states = obs_state[0]
            agent_ch_states = obs_state[1]
            ma_obs = {i: (np.append(agent_aoi_states[i], agent_ch_states[i])) for i in range(self.env.number_of_agents)}
            rew = {i: avg_reward for i in range(self.env.number_of_agents)}
            dones = {"__all__": done,}
            return ma_obs, rew, dones, info

    return MultiEnv
