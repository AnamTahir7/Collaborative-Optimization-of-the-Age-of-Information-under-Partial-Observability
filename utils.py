""" A place to store all utility functions """
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from ray.tune.logger import UnifiedLogger

def create_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def get_datetime():
    """
    Returns current data and time as e.g.: '2019-4-17_21_40_56'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_short_datetime():
    """
    Returns current data and time as e.g.: '0417_214056'
    """
    return datetime.now().strftime("%m%d_%H%M%S")

def get_date():
    """
    Returns current data and time as e.g.: '0417_214056'
    """
    return datetime.now().strftime("%m%d_%H%M%S")


def ndarray_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, np.int_):
        return int(x)
    # elif isinstance(x, Iterable) and any(isinstance(y, np.ndarray) for y in x):
    #     return [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
    return x


def recursive_conversion(d, func=ndarray_to_list):
    if isinstance(d, dict):
        return {k: recursive_conversion(v) for k, v in d.items()}
    if isinstance(d, list):
        return [recursive_conversion(v) for v in d]
    return func(d)


use_particles = True


def create_file_path(number_of_agents, config, use_true_state, use_belief_state, channel_known, use_particles,
                     use_unack_msg, dt=None, r=None, true_state_thr=None):
    if config in ('rnd', 'a1', 'thr'):
        # dt = get_short_datetime()
        script_path = Path(__file__).absolute().parent
        results_dir = script_path.joinpath('Results')
        results_dir.mkdir(exist_ok=True)
        if use_unack_msg:
            un = 'use_unack_msg'
        else:
            un = 'dont_use_unack_msg'
        results_dir = results_dir.joinpath('{}_{}_{}_{}'.format(config, channel_known, use_particles, un))
        results_dir.mkdir(exist_ok=True)
        fn = dt
        if use_true_state:
            if config == 'thr':
                fn = dt + '_true_' + r + '_true_state_thr_' + true_state_thr
            else:
                fn = dt + '_true_' + r
        elif use_belief_state:
            if config == 'thr':
                fn = dt + '_belief_' + r + '_true_state_thr_' + true_state_thr
            else:
                fn = dt + '_belief_' + r
        results_dir = results_dir.joinpath(fn)
        results_dir.mkdir(exist_ok=True)
    else:
        dt = get_short_datetime()
        script_path = Path(__file__).absolute().parent
        results_dir = script_path.joinpath('Results')
        results_dir.mkdir(exist_ok=True)
        if use_unack_msg:
            un = 'use_unack_msg'
        else:
            un = 'dont_use_unack_msg'

        results_dir = results_dir.joinpath('{}_{}_{}_{}_{}'.format(number_of_agents, config, channel_known,
                                                                          use_particles, un))
        results_dir.mkdir(exist_ok=True)
        fn = dt
        if use_true_state:
            fn = dt + '_true'
        elif use_belief_state:
            fn = dt + '_belief'
        results_dir = results_dir.joinpath(fn)
        results_dir.mkdir(exist_ok=True)
    return results_dir


def save_to_file(curr_output_json, output_dir, i):
    curr_output_json = recursive_conversion(curr_output_json)
    if isinstance(i, str):
        with open(os.path.join(output_dir, 'data_{}.json'.format(i)), 'w') as json_file:
            json.dump(curr_output_json, json_file, indent=4)
    else:
        with open(os.path.join(output_dir, 'data_{:05}.json'.format(i)), 'w') as json_file:
            json.dump(curr_output_json, json_file, indent=4)


def save_params_to_file(params: dict, path: Path):
    _params = recursive_conversion(params)
    with path.open('w') as rf:
        json.dump(_params, rf, indent=4)

def custom_log_creator(results_dir):
    def logger_creator(config):
        return UnifiedLogger(config, str(results_dir), loggers=None)

    return logger_creator


