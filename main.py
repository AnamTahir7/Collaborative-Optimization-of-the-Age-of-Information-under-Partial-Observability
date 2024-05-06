import rllib_nagent_action_ppo_ps, rllib_mfagent_action_ppo, rllib_saagent_action_ppo, constant_rate_policy, always_send,\
    threshold_policy
import time
import utils
from argparse import ArgumentParser
from aoi_env_cont_state_action import AOI_env

def run(args):
    assert(args.config in ('na_dec', 'na_cen', 'c_rate', 'a1', 'pomfc', 'thr'))
    assert(args.state in ('t','b'))
    assert(args.ch in ('known','unknown'))
    assert(args.particles in ('use','dont'))
    assert(args.drops in ('t','f'))

    number_of_agents = args.number_of_agents
    state = args.state
    config = args.config
    if args.particles == 'use':
        use_particles = True
    else:
        use_particles = False

    if args.ch == 'known':
        channel_known = True
        ch = 'channel_known'
    else:
        channel_known = False
        ch = 'channel_unknown'

    lambda_val = 0.5

    simulation_timesteps = 50
    number_particles = 100

    if state == 't':
        use_true_state = True
        use_belief_state = False
    elif state == 'b':
        use_belief_state = True
        use_true_state = False
    else:
        raise NotImplementedError

    action_space_size = 16      # {(0,1), (1,2), (2,3), ..., (15, \infty)}

    if config in ('c_rate', 'thr'):
        eval = True
        dt = args.dt
        r = args.r
        true_state_thr = args.true_state_thr
    else:
        eval = False
        dt = None
        r = None
        true_state_thr = None

    if args.drops == 't':
        drops = True
    else:
        drops = False

    results_dir = utils.create_file_path(number_of_agents, config, use_true_state, use_belief_state, ch,
                                         use_particles, drops, dt=dt, r=r, true_state_thr=true_state_thr)

    env = AOI_env(simulation_timesteps, number_particles, number_of_agents, config, results_dir, action_space_size,
                  use_true_state, use_belief_state, lambda_val, channel_known, use_particles, eval,
                  drops, r=r, true_state_thr=true_state_thr)

    def env_creator(env_config):
        return AOI_env(simulation_timesteps, number_particles, number_of_agents, config, results_dir, action_space_size,
                       use_true_state, use_belief_state, lambda_val, channel_known, use_particles, eval,
                       drops, r=r, true_state_thr=true_state_thr)

    data = {
        'lambda_val': lambda_val,
        'simulation_timesteps': simulation_timesteps,
        'number_particles': number_particles,
        'action_space_size': action_space_size,
        'number_of_agents': number_of_agents,
        'config': config,
        'use_true_state': use_true_state,
        'use_belief_state': use_belief_state,
        'use_particles': use_particles,
        'channel_known': channel_known,
        "drops": drops
    }

    utils.save_params_to_file(data, results_dir.joinpath('data.json'))

    if config in ('pomfc'):
        print("Configuration:", config)
        solver = rllib_mfagent_action_ppo.RLLibSolver(env_creator, results_dir=results_dir)
    elif config == 'na_dec':
        print("Configuration:", config)
        solver = rllib_nagent_action_ppo_ps.RLLibSolver(env_creator, results_dir=results_dir)
    elif config == 'na_cen':  # single agent centralised setup - can only be evaluated on trained agent number
        print("Configuration:", config)
        solver = rllib_saagent_action_ppo.RLLibSolver(env_creator, results_dir=results_dir)
    elif config == 'c_rate':
        solver = constant_rate_policy.ConstantRate(env_creator, results_dir=results_dir)
    elif config == 'thr':
        solver = threshold_policy.THR(env_creator, results_dir=results_dir)
    elif config == 'a1':
        solver = always_send.A1(env_creator, results_dir=results_dir)
    else:
        raise NotImplementedError
    begin = time.time()
    avg_reward, min_reward, max_reward, trainer = solver.solve(env)
    print(time.time() - begin)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('number_of_agents', type=int)
    parser.add_argument('config', type=str)  # {na_dec, na_cen, pomfc, c_rate, a1, thr}
    parser.add_argument('state', type=str)  # {t, b}
    parser.add_argument('ch', type=str)  # {known, unknown}
    parser.add_argument('particles', type=str)  # {use, dont}
    parser.add_argument('drops', type=str)  # {t, f }
    parser.add_argument('--dt', default=None)  # {t, f}
    parser.add_argument('--r', default=None)  # {0.1 ---- 1.0}, {1,2,3,...15}
    parser.add_argument('--true_state_thr', default=None)  # {t,f}

    args = parser.parse_args()

    run(args=args)