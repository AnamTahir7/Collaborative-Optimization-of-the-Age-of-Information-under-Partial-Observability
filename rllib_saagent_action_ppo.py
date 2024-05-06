import os
import shutil
import time
import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from utils import custom_log_creator

class RLLibSolver():
    """
    Approximate deterministic solutions using Rllib
    """

    def __init__(self, env_creator, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs

    def solve(self, env, **kwargs):
        ray.init(local_mode=True)
        register_env("MFenv", self.env_creator)
        trainer = ppo.PPOTrainer(
            env="MFenv",
            logger_creator=custom_log_creator(self.kwargs.get('results_dir')),
            config={
                'framework': 'tf2',
                "num_workers": 1,
                "train_batch_size": 900,
                "sgd_minibatch_size": 128,
                "num_sgd_iter": 5,
            },
        )

        logs = []
        avg_reward = []
        min_reward = []
        max_reward = []
        min_rew = None
        min_rew_thr = -0.01
        best_results_dir = self.kwargs.get('results_dir').joinpath('best_result')
        best_results_dir.mkdir(exist_ok=True)
        best_result_iter = 0
        begin = time.time()
        i = 0
        while True:
            i += 1
            print(i)
            log = trainer.train()
            time_elapsed = time.time() - begin
            print(f'step: {i}; time elapsed: {time_elapsed/60:.2f} mins')
            logs.append(log)
            if not np.isnan(log['episode_reward_mean']):
                if min_rew is None:
                    min_rew = log['episode_reward_mean']
                    trainer.save(checkpoint_dir=str(best_results_dir))
                elif min_rew - log['episode_reward_mean'] < min_rew_thr:
                    min_rew = log['episode_reward_mean']
                    shutil.rmtree(best_results_dir.joinpath(os.listdir(best_results_dir)[0]), ignore_errors=True)
                    trainer.save(checkpoint_dir=str(best_results_dir))
                    best_result_iter = trainer.get_state()['iteration']
            print('mean reward', log['episode_reward_mean'])
            avg_reward.append(log['episode_reward_mean'])
            min_reward.append(log['episode_reward_min'])
            max_reward.append(log['episode_reward_max'])

            if i % 49 == 0:
                trainer.save(checkpoint_dir=str(self.kwargs.get('results_dir')))
            if trainer.get_state()['iteration'] - best_result_iter > 200:
                trainer.save(checkpoint_dir=str(self.kwargs.get('results_dir')))
                ray.shutdown()
                break

        return avg_reward, min_reward, max_reward, trainer
