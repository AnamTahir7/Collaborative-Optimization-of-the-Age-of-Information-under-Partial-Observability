import json
import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env
import re

from ma_env_wrapper import make_multi_agent
from utils import save_to_file


class EvaluatePolicy:
    def __init__(self, trainer, results_dir, **run_parameters):
        """

        :param trainer: preloaded trainer
        :param results_dir: a directory with parameters and checkpoints
        :param run_parameters: preloaded run parameters for the environment
        """
        self.params = run_parameters
        self.trained_policy = self.params.pop('trained_policy', 'pomfc')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.trainer = trainer

    @classmethod
    def from_checkpoint(cls, checkpoint_path_orig, results_dir_suffix='eval'):

        chkp = checkpoint_path_orig.split('checkpoint_')[1]
        chk = re.search('[1-9].*$', chkp)[0]
        results_dir_suffix = results_dir_suffix+chk
        chk_complete = 'checkpoint-' + chk
        checkpoint_path = checkpoint_path_orig

        checkpoint_path = Path(checkpoint_path)
        run_params_path = checkpoint_path.parent.parent.joinpath('data.json')
        with run_params_path.open('r') as jf:
            run_params = json.load(jf)
        run_params['eval'] = True
        trained_policy = run_params.pop('config')
        global AOI_env

        with checkpoint_path.parent.parent.joinpath('params.pkl').open('rb') as pf:
            config = pickle.load(pf)

        _chkpnt_file = str(checkpoint_path.joinpath(chk_complete))

        config['env_config'] = run_params
        config['num_workers'] = 1
        config['env_config']['config'] = trained_policy
        config['disable_env_checking'] = True
        ray.init(local_mode=False)
        if trained_policy in ('mf', 'na_cen', 'pomfc'):  # single agent
            register_env("EVAL", lambda x: cls.create_env(x))
            agent = ppo.PPOTrainer(config, env='EVAL')
        else:
            ma_env_cls = make_multi_agent(cls.create_env)
            agent = ppo.PPOTrainer(config, env=ma_env_cls)

        agent.restore(_chkpnt_file)
        run_params['trained_policy'] = trained_policy
        return cls(agent, checkpoint_path.parent.joinpath(results_dir_suffix), **run_params)

    @staticmethod
    def create_env(params):
        env = AOI_env(**params)
        return env

    def single_run_test(self, env, number_of_agents_test):
        time_steps = self.params.get('simulation_timesteps', 50)
        obs = env.reset()
        action_trajectory = []
        for i in range(time_steps):
            if self.trained_policy == 'pomfc':
                action_orig = self.trainer.compute_single_action(obs)
                action = action_orig
            elif self.trained_policy == 'na_cen':
                action = self.trainer.compute_single_action(obs)
            elif self.trained_policy == 'na_dec':
                action = []
                agent_states = obs[0]
                agent_ch_states = obs[1]
                for n in range(number_of_agents_test):
                    obss = np.append(agent_states[n], agent_ch_states[n])
                    if self.trained_policy == 'na_dec':
                        action_per_policy = self.trainer.compute_single_action(obss, policy_id='shared_policy')
                    else:
                        action_per_policy = self.trainer.compute_single_action(obss, policy_id='policy_{}'.format(n))
                    action.append(action_per_policy)
            else:
                raise NotImplementedError(f'Policy is not implemented for {self.params.get("trained_policy")}')

            action_trajectory.append(action)
            obs, joint_reward_np, _, _ = env.step(action)

        # evaluate on the trajectory
        all_rewards = []
        all_drops = []
        all_aoi = np.zeros([time_steps, env.number_of_agents])
        obs = env.reset()
        for i in range(time_steps):
            action = action_trajectory[i]
            obs, joint_reward_np, _, _ = env.step(action)
            all_rewards.append(joint_reward_np[0])
            all_drops.append(joint_reward_np[1])
            all_aoi[i] = joint_reward_np[2]
        cum_reward = np.cumsum(all_rewards)
        return cum_reward, all_rewards, all_aoi, all_drops

    def run_test(self, no_agents_test_curr):
        no_mc = 100
        print(no_agents_test_curr)
        sum_ep_rewards = np.zeros(self.params.get('simulation_timesteps'))
        cum_reward_per_ep = np.zeros(self.params.get('simulation_timesteps'))

        file_path = 'N_{}'.format(no_agents_test_curr)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(parents=True,exist_ok=True)

        current_params = deepcopy(self.params)
        current_params['number_of_agents'] = no_agents_test_curr
        current_params['results_dir'] = agent_results_dir

        test_env = self.create_env(current_params)

        # output directory
        for i in range(no_mc):
            print(i)
            cum_reward, ep_rewards, all_aoi, all_drops = self.single_run_test(test_env, no_agents_test_curr)
            sum_ep_rewards += ep_rewards
            cum_reward_per_ep += cum_reward
            curr_output_json = {"no_agents": int(no_agents_test_curr),
                                "reward": ep_rewards,
                                "aoi": all_aoi,
                                "drops": all_drops
                                }
            # saving to file
            save_to_file(curr_output_json, agent_results_dir, i)

        # Take average of cum_reward over number of mc sims
        avg_cum_ep_reward = cum_reward_per_ep / no_mc

        plt.plot(np.arange(len(avg_cum_ep_reward)), avg_cum_ep_reward, 'g', label='Avg Cumulative reward')
        plt.title('{a} agents'.format(a=no_agents_test_curr))
        plt.legend()
        plt.savefig(agent_results_dir.joinpath('{}agents_avg_cum_reward_testing.pdf'.format(no_agents_test_curr)))
        plt.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('number_of_agents', type=int)
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    results_dir_suffix = 'Traj_Evaluation_checkpoint'

    current_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.path
    ep = EvaluatePolicy.from_checkpoint(output_dir, results_dir_suffix=results_dir_suffix)
    ep.run_test(args.number_of_agents)
