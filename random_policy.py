import numpy as np
from utils import save_to_file
import matplotlib.pyplot as plt

class RND:
    def __init__(self, env_creator, rnn, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs
        self.results_dir = self.kwargs.get('results_dir')
        self.run_params = {}

    def solve(self, env):
        file_path = 'N_{}'.format(env.number_of_agents)
        output_dir = self.results_dir.joinpath(file_path)
        output_dir.mkdir(exist_ok=True)
        agent_results_dir = output_dir
        episode_timesteps = env.episode_timesteps
        mc = 100
        cum_reward_per_ep = np.zeros(episode_timesteps)
        self.rate = int(np.ceil(env.r * env.number_of_agents))
        for j in range(mc):
            print(j)
            all_rewards = []
            all_drops = []
            all_aoi = np.zeros([episode_timesteps, env.number_of_agents])
            sum_ep_rewards = np.zeros(episode_timesteps)
            curr_obs = env.reset()
            for i in range(episode_timesteps):
                action_all_agents = self.get_action(env)
                curr_obs, joint_reward, _, _ = env.step(action_all_agents)
                all_rewards.append(joint_reward[0])
                all_drops.append(joint_reward[1])
                all_aoi[i] = joint_reward[2]
            cum_reward = np.cumsum(all_rewards)
            sum_ep_rewards += all_rewards
            cum_reward_per_ep += cum_reward
            curr_output_json = {"no_agents": int(env.number_of_agents),
                                "reward": all_rewards,
                                "aoi": all_aoi,
                                "drops": all_drops
                                }
            # saving to file
            save_to_file(curr_output_json, agent_results_dir, j)

        # Take average of cum_reward over number of mc sims
        avg_cum_ep_reward = cum_reward_per_ep / mc

        plt.plot(np.arange(len(avg_cum_ep_reward)), avg_cum_ep_reward, 'g', label='Avg Cumulative reward')
        plt.title('{a} agents'.format(a=env.number_of_agents))
        plt.legend()
        plt.savefig(agent_results_dir.joinpath('{}agents_avg_cum_reward_testing.pdf'.format(env.number_of_agents)))
        plt.close()

        return 0, 0, 0, 0

    def get_action(self, env):    # action for every agent
        action0 = np.zeros(env.number_of_agents)
        action0[np.random.choice(env.number_of_agents, size=self.rate,replace=False)] = 1
        action1 = np.ones(env.number_of_agents)
        actions = [action0, action1]
        return actions

