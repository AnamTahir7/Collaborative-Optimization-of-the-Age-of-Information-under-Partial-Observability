import gym
import numpy as np
from gym import spaces
from bisect import bisect
import portion as P

class AOI_env(gym.Env):
    def __init__(self, simulation_timesteps=30, number_particles=100, number_of_agents=2, config='mf', results_dir=None,
                 action_space_size=2, weighted_action=False, use_true_state=False, use_belief_state=False,
                 lambda_val=1, channel_known=False, use_particles=False, eval=False, use_unack_msg=False):
        self.use_unack_msg = use_unack_msg
        self.eval = eval
        self.use_particles = use_particles
        self.channel_known = channel_known
        self.use_true_state = use_true_state
        self.use_belief_state = use_belief_state
        self.weighted_action = weighted_action
        self.results_dir = results_dir
        self.number_of_particles = number_particles
        self.number_of_agents = number_of_agents
        self.config = config
        self.done = False
        self.infos = {}
        self.episode_timesteps = simulation_timesteps
        self.max_episode_steps = simulation_timesteps
        self.max_to_sender_channel_capacity = 50 * self.number_of_agents  # infinite capacity delayed channel to send back acks
        self.max_initial_timeslot = 2
        self.max_discrete_delay = 5
        self.lambda_val = lambda_val
        self.max_to_rx_channel_capacity = int(np.ceil(self.number_of_agents / 2))
        self.plotting_time = 50
        self.obs_size = 6  # 3 mean and 3 std
        self.max_x_t_obs = simulation_timesteps
        self.max_time = 3
        self.action_space_size = action_space_size
        if self.weighted_action:
            self.mid_space = np.arange(self.action_space_size) + 0.5
            self.sigma = 2
        obs_high = max(self.max_x_t_obs, self.max_to_rx_channel_capacity+1)
        if self.config == 'mf':
            self.action_space = spaces.Tuple((spaces.MultiDiscrete([2] * self.action_space_size),
                                              spaces.MultiDiscrete([self.max_time] * self.action_space_size)))
            self.observation_space = spaces.Box(low=0, high=obs_high, shape=(self.obs_size+1,),
                                                dtype=np.float32)
        elif self.config in ('na_dec'):
            self.action_space = spaces.Tuple((spaces.MultiDiscrete([2] * self.action_space_size),
                                              spaces.MultiDiscrete([self.max_time] * self.action_space_size)))
            if self.use_true_state:
                self.observation_space = spaces.Box(low=0, high=obs_high, shape=(3,), dtype=np.float32)
            else:
                if self.use_particles:
                    self.observation_space = spaces.Box(low=0, high=obs_high, shape=(self.number_of_particles+2,),
                                                        dtype=np.float32)
                else:
                    self.observation_space = spaces.Box(low=0, high=obs_high, shape=(4,), dtype=np.float32)
        elif self.config == 'na_cen':  # single agent model
            self.action_space = spaces.Tuple((spaces.MultiDiscrete([2] * self.action_space_size),
                                              spaces.MultiDiscrete([self.max_time] * self.action_space_size)))
            if self.use_true_state:
                self.observation_space = spaces.Tuple((spaces.Box(low=0, high=obs_high, shape=(self.number_of_agents,),
                                                                  dtype=np.float32),
                                                       spaces.Box(low=0, high=self.max_to_rx_channel_capacity, shape=(self.number_of_agents,),
                                                                   dtype=np.float32)))
            else:
                if self.use_particles:
                    self.observation_space = spaces.Tuple((spaces.Box(low=0, high=obs_high, shape=(self.number_of_agents, self.number_of_particles),
                                                                  dtype=np.float32),
                                                       spaces.Box(low=0, high=self.max_to_rx_channel_capacity, shape=(self.number_of_agents,),
                                                                   dtype=np.float32)))
                else:
                    self.observation_space = spaces.Tuple((spaces.Box(low=0, high=obs_high, shape=(self.number_of_agents,), #mean
                                                                  dtype=np.float32),
                                                           spaces.Box(low=0, high=obs_high,     #std
                                                                       shape=(self.number_of_agents,),
                                                                       dtype=np.float32),
                                                       spaces.Box(low=0, high=self.max_to_rx_channel_capacity,
                                                                  shape=(self.number_of_agents,),
                                                                  dtype=np.float32)))
        else:
            raise NotImplementedError

    def reset(self):
        '''
        :return: empty channel - state 0
                and reset time
        '''
        self.dropped_msgs_each_time = 0
        self.all_particles_each_agent = [[[[0], [0]] for _ in range(self.number_of_particles)] for n in range(self.number_of_agents)]
        self.x_t_each_agent = [[0] for n in range(self.number_of_agents)]
        self.m_each_agent = [None for n in range(self.number_of_agents)]
        self.unack_msg_idx_each_agent = [[] for n in range(self.number_of_agents)]
        self.tau_m_idx_each_agent = [0 for n in range(self.number_of_agents)]  # time when msg is sent to rx
        self.tau_m_prime_idx_each_agent = [0 for n in range(self.number_of_agents)]  # time when msg rcvd by rx
        self.obs_times_idx_each_agent = [0 for n in range(self.number_of_agents)]
        self.msg_idx_sent_to_rx_each_agent = [[] for n in range(self.number_of_agents)]
        self.msg_idx_rcvd_by_rx_each_agent = [[] for n in range(self.number_of_agents)]
        self.msg_idx_rcvd_by_sender_each_agent = [[] for n in range(self.number_of_agents)]
        self.obsolete_msg_idx_each_agent = [[] for n in range(self.number_of_agents)]
        self.sorted_tau_m_prime_idx_each_agent = [[] for n in range(self.number_of_agents)]
        self.obs_times_each_agent = [[None] for n in range(self.number_of_agents)]
        self.sorted_tau_m_prime_each_agent = [[None] for n in range(self.number_of_agents)]
        self.tau_m_each_agent = [[None] for n in range(self.number_of_agents)]
        self.delay_sent_message_each_agent = [[None] for n in range(self.number_of_agents)]
        self.sorted_obs_times_each_agent = [[None] for n in range(self.number_of_agents)]
        self.state_each_agent = np.zeros([self.number_of_agents, 4])
        self.channel_state_to_rx_each_agent = [{} for n in range(self.number_of_agents)]  # status of each job in the channel
        self.channel_state_to_sender_each_agent = [{} for n in range(self.number_of_agents)]  # status of each job in the channel
        self.total_jobs_in_to_sender_channel = 0
        self.total_jobs_in_to_rx_channel = 0
        self.all_events = []
        self.all_event_agents = []
        self.time_each_agent = [[0] for n in range(self.number_of_agents)]
        self.last_aoi_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.ack_info_each_agent = [{} for n in range(self.number_of_agents)]
        self.simulated_delays_each_particle_each_agent = []
        self.simulated_delays_to_sample_from = np.random.poisson(self.lambda_val, size=1000) + 1 # only discrete delays
        self.simulated_delays_to_sample_from[self.simulated_delays_to_sample_from > 3] = 3
        for i in range(self.number_of_agents):
            self.simulated_delays_each_particle_each_agent.append(self.simulated_delays_to_sample_from)
        self.done = False
        self.curr_time = 0
        self.prev_time = 0
        self.time_to_send_each_packet = np.random.randint(self.max_initial_timeslot, size=self.number_of_agents, dtype=int)
        self.delay_to_rx_each_agent_each_job = [[] for n in range(self.number_of_agents)]
        self.channel_state_obs_each_agent = [self.total_jobs_in_to_rx_channel] * self.number_of_agents
        self.mean_channel_state = self.total_jobs_in_to_rx_channel
        self.last_ch_obs_time_each_agent = np.zeros(self.number_of_agents, dtype=np.float32)
        if self.config == 'mf':
            empty_obs = np.zeros((self.obs_size, 1), dtype=np.float32)
            curr_obs = np.append(empty_obs, self.mean_channel_state)
        elif self.config in ('na_dec'):
            # y, no of unack msgs for each agent and channel state
            if self.use_true_state:
                curr_obs = np.zeros((self.number_of_agents, 2), dtype=np.float32), self.last_ch_obs_time_each_agent
            else:
                if self.use_particles:
                    curr_obs = np.zeros((self.number_of_agents, self.number_of_particles+1), dtype=np.float32), self.last_ch_obs_time_each_agent
                else:
                    curr_obs = np.zeros((self.number_of_agents, 3), dtype=np.float32), self.last_ch_obs_time_each_agent
        elif self.config == 'na_cen':
            if self.use_true_state:
                curr_obs = np.zeros((self.number_of_agents), dtype=np.float32), self.last_ch_obs_time_each_agent
            else:
                if self.use_particles:
                    curr_obs = np.zeros((self.number_of_agents, self.number_of_particles), dtype=np.float32), self.last_ch_obs_time_each_agent
                else:
                    curr_obs = np.zeros((self.number_of_agents), dtype=np.float32), np.zeros((self.number_of_agents),
                                                                dtype=np.float32), self.last_ch_obs_time_each_agent
        else:
            raise NotImplementedError
        self.division = P.IntervalDict()
        self.division[P.closed(0, 1)] = 0
        self.division[P.closed(1, 2)] = 1
        self.division[P.closed(2, 3)] = 2
        self.division[P.closed(3, 4)] = 3
        self.division[P.closed(4, 5)] = 4
        self.division[P.closed(5, 6)] = 5
        self.division[P.closed(6, 7)] = 6
        self.division[P.closed(7, 8)] = 7
        self.division[P.closed(8, 9)] = 8
        self.division[P.closed(9, 10)] = 9
        self.division[P.closed(10, 11)] = 10
        self.division[P.closed(11, 12)] = 11
        self.division[P.closed(12, 13)] = 12
        self.division[P.closed(13, 14)] = 13
        self.division[P.closed(14, 15)] = 14
        self.division[P.closed(15, 100)] = 15
        if self.use_unack_msg:
            self.division2 = P.IntervalDict()
            self.division2[P.closed(0, 0.1)] = 0
            self.division2[P.closed(0.1, 0.2)] = 1
            self.division2[P.closed(0.2, 0.3)] = 2
            self.division2[P.closed(0.3, 0.4)] = 3
            self.division2[P.closed(0.4, 0.5)] = 4
            self.division2[P.closed(0.5, 0.6)] = 5
            self.division2[P.closed(0.6, 0.7)] = 6
            self.division2[P.closed(0.7, 0.8)] = 7
            self.division2[P.closed(0.8, 0.9)] = 8
            self.division2[P.closed(0.9, 1.0)] = 9
        self.last_when_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.last_rcvd_obs_each_agent = [0 for n in range(self.number_of_agents)]
        self.last_aoi_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.expected_belief_x_t_each_agent = [0] * self.number_of_agents
        self.std_belief_x_t_each_agent = [0] * self.number_of_agents
        self.curr_particles_each_agent = [[0] * self.number_of_particles] * self.number_of_agents
        return curr_obs

    def step(self, action_list):
        action_each_agent, time_till_next_contol_each_agent = self.extract_actions(action_list)
        reward, mean_obs = self.simulate(action_each_agent, time_till_next_contol_each_agent)
        if self.curr_time >= self.episode_timesteps - 1:
            self.done = True
            self.infos["observation"] = mean_obs

        if self.channel_known:
            self.channel_state_obs_each_agent = [self.total_jobs_in_to_rx_channel] * self.number_of_agents
            self.mean_channel_state = self.total_jobs_in_to_rx_channel
        else:
            self.channel_state_obs_each_agent = self.last_ch_obs_time_each_agent
            self.mean_channel_state = np.mean(self.last_ch_obs_time_each_agent)

        self.curr_particles_each_agent = []
        self.expected_belief_x_t_each_agent = []
        self.std_belief_x_t_each_agent = []
        for n in range(self.number_of_agents):
            particles = self.all_particles_each_agent[n]
            expected_belief, std_belief = self.calculate_belief_from_particles(particles, self.curr_time)
            self.expected_belief_x_t_each_agent.append(expected_belief)
            self.std_belief_x_t_each_agent.append(std_belief)

        if self.config == 'mf':
            if self.use_true_state:
                curr_obs = np.append(mean_obs, self.mean_channel_state)
            else:
                mean_belief = np.mean(self.expected_belief_x_t_each_agent)
                std_belief = np.std(self.expected_belief_x_t_each_agent)
                mean_obs[0] = mean_belief
                mean_obs[3] = std_belief
                curr_obs = np.append(mean_obs, self.mean_channel_state)            # curr_obs = np.append(mean_obs, self.total_jobs_in_to_rx_channel)
        elif self.config in ('na_dec'):
            if self.use_true_state:
                obs = np.zeros((self.number_of_agents, 2))
                obs[:, 0] = self.state_each_agent[:, 0]   # true aoi
            else:
                if self.use_particles:
                    obs = np.zeros((self.number_of_agents, self.number_of_particles+1))
                    obs[:, 0:self.number_of_particles] = self.curr_particles_each_agent
                else:
                    obs = np.zeros((self.number_of_agents, 3))
                    obs[:, 0] = self.expected_belief_x_t_each_agent  # belief over aoi
                    obs[:, 1] = self.std_belief_x_t_each_agent  # belief over aoi
            obs[:, -1] = np.sum(self.state_each_agent[:, 1:3], axis=1)
            curr_obs = obs, self.channel_state_obs_each_agent
        elif self.config == 'na_cen':
            if self.use_true_state:
                curr_obs = self.state_each_agent[:, 0], np.array(self.channel_state_obs_each_agent)
            else:
                if self.use_particles:
                    curr_obs = self.curr_particles_each_agent, self.channel_state_obs_each_agent
                else:
                    curr_obs = np.array(self.expected_belief_x_t_each_agent, dtype=np.float32), \
                        np.array(self.std_belief_x_t_each_agent,dtype=np.float32),  \
                                self.channel_state_obs_each_agent
        else:
            raise NotImplementedError
        # ch obs should be 0 for those who did not receive any obs
        self.last_ch_obs_time_each_agent = np.zeros(self.number_of_agents, dtype=np.float32)
        if self.eval:
            reward = reward, self.dropped_msgs_each_time, self.state_each_agent[:, 0]
        self.prev_time = self.curr_time
        self.curr_time += 1     # working in discrete time steps only
        return curr_obs, reward, self.done, self.infos

    def calculate_belief_from_particles(self, particles, time):
        curr_x_all = []
        for t, x in particles:
            try:
                idx_time = t.index(time)
            except:
                print('Idx not found')
                raise IndexError
            else:
                curr_x_all.append(x[idx_time])
        self.curr_particles_each_agent.append(curr_x_all)
        expected_belief = np.mean(curr_x_all)
        std_belief = np.std(curr_x_all)
        return expected_belief, std_belief

    def extract_actions(self, action_list):
        action_each_agent = []
        time_till_next_control_each_agent = []
        all_idx = []
        if self.weighted_action:
            if self.use_true_state:
                for n in range(self.number_of_agents):
                    r = self.mid_space - self.x_t_each_agent[n][-1]
                    r_squared = -np.square(r)
                    expon_weights = np.exp(r_squared/(2 * (self.sigma**2)))
                    normalised_weights = expon_weights/sum(expon_weights)
                    weighted_actions = normalised_weights * action_list[n]
                    action_each_agent.append(np.round(sum(weighted_actions[0])))
                    time_till_next_control_each_agent.append(sum(weighted_actions[1]))
            else:
                for n in range(self.number_of_agents):
                    r = self.mid_space - self.expected_belief_x_t_each_agent[n]
                    r_squared = -np.square(r)
                    expon_weights = np.exp(r_squared/(2 * (self.sigma**2)))
                    normalised_weights = expon_weights/sum(expon_weights)
                    weighted_actions = normalised_weights * action_list[n]
                    action_each_agent.append(np.round(sum(weighted_actions[0])))
                    time_till_next_control_each_agent.append(sum(weighted_actions[1]))
        else:
            for n in range(self.number_of_agents):
                all_idx.append(self.division[self.expected_belief_x_t_each_agent[n]])
            if self.config == 'mf':
                time_till_next_control_each_agent = action_list[1][all_idx]
                action_each_agent = action_list[0][all_idx]
            elif self.config in ('na_dec'):
                for n in range(self.number_of_agents):
                    time_till_next_control_each_agent.append(action_list[n][1][all_idx[n]])
                    action_each_agent.append(action_list[n][0][all_idx[n]])
            elif self.config == 'na_cen':
                for n in range(self.number_of_agents):
                    time_till_next_control_each_agent.append(action_list[1][all_idx[n]])
                    action_each_agent.append(action_list[0][all_idx[n]])
            else:
                y_t_each_agent = self.state_each_agent[:, 3]
                all_idx = []
                for n in range(self.number_of_agents):
                    all_idx.append(self.division[y_t_each_agent[n]])
                time_till_next_control_each_agent = action_list[1][all_idx]
                action_each_agent = action_list[0][all_idx]

        time_till_next_control_each_agent = np.array(time_till_next_control_each_agent)
        time_till_next_control_each_agent[time_till_next_control_each_agent == 0] = 1

        return action_each_agent, time_till_next_control_each_agent
    def simulate_all_particles_until_obs_received(self, prev_time, obs_time, simulated_delay_each_particle, particle,
                                                  tot_unck_msgs_particle_sim, back_prop=False, x_rho=None):
        """
        simuate particles last observation time till next one based on ho wmany unacked msgs are in the channel/rx
        :param prev_time: time when last observation was rcvd at sender or 0 at start
        :param obs_time: current obsrevation time
        :param simulated_delay_each_particle: channel delay for each particle
        :param particle: set of particles to be updated
        :param tot_unck_msgs_particle_sim: total msgs for which acks are still missing
        :param back_prop: True when an observation is rcvd
        :param x_rho: value of aoi rcvd by the sender in the ack
        :return:
        simulates all particles till the current observation time
        """
        prev_time_sim = prev_time.copy()
        if back_prop:  # start from data recevied in obs/ack
            t = prev_time_sim.copy()
            x_t = x_rho.copy()
        else:
            t = []
            x_t = []
            for i in range(len(particle)):  # start from last time and aoi of each particle
                t.append(particle[i][0][-1])
                x_t.append(particle[i][1][-1])
            t = np.array(t, dtype=float)
            x_t = np.array(x_t, dtype=float)
        particle_with_no_new_msg = []
        particle_with_new_msg = []
        delays_non_obs_msgs_all_particles = [[] for _ in range(self.number_of_particles)]
        if tot_unck_msgs_particle_sim > 0:  # if there are any unacked msgs in channel then simulate them
            for i in range(self.number_of_particles):
                delays_unack_msgs = np.random.choice(simulated_delay_each_particle, size=tot_unck_msgs_particle_sim,
                                                     replace=False)
                tau_m_prime_sim_unack_msgs = t[i] + np.cumsum(delays_unack_msgs)
                # check which of the unacked msgs could be rcvd within the next observation time
                possible_rcvd_acks = np.where(tau_m_prime_sim_unack_msgs <= obs_time)[0].tolist()
                if len(possible_rcvd_acks) > 1:
                    delay_non_obsolete_m_particles, eta_j_particles = self.get_non_obsolete_messages(
                        tau_m_prime_sim_unack_msgs[possible_rcvd_acks], delays_unack_msgs[possible_rcvd_acks])
                    obs_idx = []
                    for kk in range(len(delays_unack_msgs)):  # get idx of obsolete messages
                        if delays_unack_msgs[kk] not in delay_non_obsolete_m_particles:
                            obs_idx.append(kk)
                    # remove obsolete
                    if len(obs_idx) > 0:  # remove obsolete msgs since their acks will not be rcvd
                        for o in obs_idx:
                            if o in possible_rcvd_acks:
                                possible_rcvd_acks.remove(o)
                    delays_non_obs_msgs_all_particles[i] = delays_unack_msgs[possible_rcvd_acks].tolist()
                    particle_with_new_msg.append(i)
                elif len(possible_rcvd_acks) == 1:
                    particle_with_new_msg.append(i)
                    delays_non_obs_msgs_all_particles[i].append(delays_unack_msgs[possible_rcvd_acks[0]])
                else:
                    particle_with_no_new_msg.append(i)  # for these simulate directly till obs_time w/o any acks
        else:  # no unacked msgs in the channel so no acks expected
            particle_with_no_new_msg = np.arange(self.number_of_particles)

        if len(particle_with_no_new_msg) > 0:  # simulate w/o any acks till observation time
            for m in particle_with_no_new_msg:
                t[m] = obs_time
                x_t[m] += (obs_time - prev_time_sim[m])
                particle[m][0].append(t[m])
                particle[m][1].append(x_t[m])
                prev_time_sim[m] = obs_time

        if len(particle_with_new_msg) > 0:  # simulate with acks
            for m in particle_with_new_msg:
                for delay in delays_non_obs_msgs_all_particles[m]:
                    t[m] += delay
                    x_t[m] += delay
                    particle[m][0].append(t[m])
                    particle[m][1].append(x_t[m])
                    x_t[m] -= delay
                    particle[m][0].append(t[m])
                    particle[m][1].append(x_t[m])
                diffr = obs_time - t[m]  # at the end also simulate from prev_time till obs_time w/o acks
                t[m] = obs_time
                if x_t[m] + diffr < 0:
                    x_t[m] = 0
                else:
                    x_t[m] += diffr
                particle[m][0].append(t[m])
                particle[m][1].append(x_t[m])
                prev_time_sim[m] = obs_time

    def get_non_obsolete_messages(self, tau_m_prime, delay_each_sent_message):
        """
        get the order in which messages are rcvd at the rx and then using only the non-obsolete ones to update aoi and
        send back acks of
        :param tau_m_prime: unordered rcvng times
        :param delay_each_sent_message: delay faced by each msg in the channel
        :return:
        delay_non_obsolete_m: delay of only non obsolete messgaes, which are used to update the aoi x_t
        eta_j: msg rcving time at rx of only non obsolete msgs
        """
        orig_sorted_tau_prime_idx = np.argsort(tau_m_prime)
        sorted_tau_m_prime = tau_m_prime[orig_sorted_tau_prime_idx]
        eta_idx = []
        eta = []
        eta_idx.append(orig_sorted_tau_prime_idx[0])
        eta.append(sorted_tau_m_prime[0])
        last_max_idx = orig_sorted_tau_prime_idx[0]
        for i in range(1, len(orig_sorted_tau_prime_idx)):
            if orig_sorted_tau_prime_idx[i] > orig_sorted_tau_prime_idx[i - 1] and \
                    orig_sorted_tau_prime_idx[i] > last_max_idx:
                eta_idx.append(orig_sorted_tau_prime_idx[i])
                eta.append(sorted_tau_m_prime[i])
                last_max_idx = orig_sorted_tau_prime_idx[i]
        assert sorted(eta_idx) == eta_idx
        non_obsolete_m = eta_idx
        delay_non_obsolete_m = delay_each_sent_message[non_obsolete_m]
        eta_j = np.array(eta)
        return delay_non_obsolete_m, eta_j

    def simulate(self, action_each_agent, time_till_next_control_each_agent):
        '''
        return reward, next state and obs together from this simulator for N agent case
        :param action:
        :param curr_state:
        :return: total reward to all for now, can also be factored, next state and obs
        '''

        rx_rcvd_msg = []
        agents_rcvd_acks = []
        if self.curr_time != 0:  # update x_t and particles based on acks or msg rcvd or not
            for n in range(self.number_of_agents):
                if self.sorted_obs_times_each_agent[n][0] is not None:
                    if len(np.where(np.array(self.sorted_obs_times_each_agent[n]) == self.curr_time)[0]) > 0:
                        agents_rcvd_acks.append(n)
                if self.sorted_tau_m_prime_each_agent[n][0] is not None:
                    if len(np.where(np.array(self.sorted_tau_m_prime_each_agent[n]) == self.curr_time)[0]) > 0:
                        rx_rcvd_msg.append(n)

            # for agents not recieving any obs at this time, should be assigned to 0
            for n in range(self.number_of_agents):
                if n not in agents_rcvd_acks:
                    self.state_each_agent[n][3] = 0

            if rx_rcvd_msg:  # update true x_t of agnets and particle update as before
                for event_agent in rx_rcvd_msg:
                    # check how many are observed in this timeslot, could be more than 1
                    arr_times = (np.array(self.sorted_tau_m_prime_each_agent[event_agent]))
                    total_msgs_arriving = len(np.where((arr_times == self.curr_time) & (arr_times > self.prev_time))[0])
                    for a in range(total_msgs_arriving):
                        time_past = self.curr_time - self.prev_time
                        rcvd_msg_idx = self.sorted_tau_m_prime_idx_each_agent[event_agent][
                            self.tau_m_prime_idx_each_agent[event_agent]]
                        if len(self.msg_idx_rcvd_by_rx_each_agent[event_agent]) == 0 or \
                                sum(rcvd_msg_idx > np.array(self.msg_idx_rcvd_by_rx_each_agent[event_agent])) == \
                                len(self.msg_idx_rcvd_by_rx_each_agent[event_agent]):
                            self.msg_idx_rcvd_by_rx_each_agent[event_agent].append(rcvd_msg_idx)
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.time_each_agent[event_agent].append(self.curr_time)
                            self.x_t_each_agent[event_agent].append(
                                self.delay_sent_message_each_agent[event_agent][rcvd_msg_idx])
                            self.time_each_agent[event_agent].append(self.curr_time)
                            # simulate its observation time and update obs_time
                            delay = np.random.choice(self.simulated_delays_to_sample_from)
                            eta_prime = self.curr_time + delay
                            if None in self.obs_times_each_agent[event_agent]:
                                self.obs_times_each_agent[event_agent].remove(None)
                            self.obs_times_each_agent[event_agent].append(eta_prime)
                            sorted_obs_times_idx_each_agent = np.argsort(self.obs_times_each_agent[event_agent])
                            self.sorted_obs_times_each_agent[event_agent] = \
                                np.array(self.obs_times_each_agent[event_agent])[
                                    sorted_obs_times_idx_each_agent]
                            self.ack_info_each_agent[event_agent][eta_prime] = [self.curr_time,
                                                                                self.x_t_each_agent[event_agent][-1]]
                            self.channel_state_to_sender_each_agent[event_agent][eta_prime] = [
                                self.channel_state_to_rx_each_agent[event_agent][self.curr_time][1], self.curr_time]
                            self.total_jobs_in_to_sender_channel += 1
                            self.state_each_agent[event_agent][2] += 1
                        else:
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.obsolete_msg_idx_each_agent[event_agent].append(rcvd_msg_idx)
                            self.time_each_agent[event_agent].append(self.curr_time)
                        self.tau_m_prime_idx_each_agent[event_agent] += 1
                        self.total_jobs_in_to_rx_channel -= 1
                        self.state_each_agent[event_agent][1] -= 1
            if agents_rcvd_acks:
                for event_agent in agents_rcvd_acks:
                    # check how many are observed in this timeslot, could be more than 1
                    arr_times = (np.array(self.sorted_obs_times_each_agent[event_agent]))
                    total_msgs_arriving = len(np.where((arr_times == self.curr_time) & (arr_times > self.prev_time))[0])
                    for a in range(total_msgs_arriving):
                        self.msg_idx_rcvd_by_sender_each_agent[event_agent].append(
                            self.obs_times_idx_each_agent[event_agent])
                        rho = self.ack_info_each_agent[event_agent][self.curr_time][0]
                        x_rho = self.ack_info_each_agent[event_agent][self.curr_time][1]  # rcvd observation
                        self.last_when_obs_time_each_agent[event_agent] = self.curr_time
                        self.last_rcvd_obs_each_agent[event_agent] = x_rho
                        self.last_aoi_obs_time_each_agent[event_agent] = rho
                        self.state_each_agent[event_agent][3] = x_rho
                        prev_time = np.array([rho] * self.number_of_particles).astype(float)
                        x_rho_all = np.array([x_rho] * self.number_of_particles).astype(float)
                        self.unack_msg_idx_each_agent[event_agent].remove(self.obs_times_idx_each_agent[event_agent])
                        virtual_particles = [[[rho], [x_rho]] for _ in range(self.number_of_particles)]
                        tot_unack_msgs_particles_n = len(self.unack_msg_idx_each_agent[event_agent])
                        self.simulate_all_particles_until_obs_received(prev_time, self.curr_time,
                                                                       self.simulated_delays_each_particle_each_agent[
                                                                           event_agent],
                                                                       virtual_particles, tot_unack_msgs_particles_n,
                                                                       back_prop=True, x_rho=x_rho_all)
                        for k in range(self.number_of_particles):  # update jumps after back propagation
                            self.all_particles_each_agent[event_agent][k][1][-1] = virtual_particles[k][1][-1]
                            self.all_particles_each_agent[event_agent][k][0][-1] = virtual_particles[k][0][-1]
                        self.obs_times_idx_each_agent[event_agent] += 1
                        self.total_jobs_in_to_sender_channel -= 1
                        self.state_each_agent[event_agent][2] -= 1
                        if event_agent not in rx_rcvd_msg:
                            time_past = self.curr_time - self.prev_time
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.time_each_agent[event_agent].append(self.curr_time)

        if self.curr_time > 0:
            for n in range(self.number_of_agents):
                if n not in rx_rcvd_msg and n not in agents_rcvd_acks:  # if 1 and 2 event did not occur
                    time_past = self.curr_time - self.prev_time
                    self.x_t_each_agent[n].append(self.x_t_each_agent[n][-1] + time_past)
                    self.time_each_agent[n].append(self.curr_time)
                self.state_each_agent[n, 0] = self.x_t_each_agent[n][-1]

        dropped_msgs = 0
        if self.curr_time > 0:
            for n in range(self.number_of_agents):  # simulate till curr_t
                tot_unack_msgs_particles_n = len(self.unack_msg_idx_each_agent[n])
                x_rho = self.last_rcvd_obs_each_agent[n]
                x_rho_all = np.array([x_rho] * self.number_of_particles).astype(float)
                rho = self.last_aoi_obs_time_each_agent[n]
                virtual_particles = [[[rho], [x_rho]] for _ in range(self.number_of_particles)]
                prev_time_arr = np.array([rho] * self.number_of_particles).astype(float)
                self.simulate_all_particles_until_obs_received(prev_time_arr, self.curr_time,
                                                               self.simulated_delays_each_particle_each_agent[n],
                                                               virtual_particles,
                                                               # self.all_particles_each_agent[n],
                                                               tot_unack_msgs_particles_n,
                                                               back_prop=True, x_rho=x_rho_all)
                for k in range(self.number_of_particles):  # update jumps after back propagation
                    self.all_particles_each_agent[n][k][1].append(virtual_particles[k][1][-1])
                    self.all_particles_each_agent[n][k][0].append(virtual_particles[k][0][-1])

            msg_sending_agents_at_curr_time = np.where(np.array(action_each_agent) > 0)[0]
            agents_should_msg_at_curr_time = np.where(np.array(self.time_to_send_each_packet) == self.curr_time)[0]
            agents_sending_msg_at_curr_time = np.intersect1d(msg_sending_agents_at_curr_time, agents_should_msg_at_curr_time)
            if len(agents_sending_msg_at_curr_time) > 0:  # event == 0
                remaining_slots_in_channel = self.max_to_rx_channel_capacity - self.total_jobs_in_to_rx_channel
                total_arr_jobs = len(agents_sending_msg_at_curr_time)
                if total_arr_jobs > remaining_slots_in_channel:
                    # choose agents randomly to drop jobs
                    agents_sending_msgs_successfully = np.random.choice(agents_sending_msg_at_curr_time,
                                                                        size=remaining_slots_in_channel,
                                                                        replace=False)
                    dropped_msgs = total_arr_jobs - remaining_slots_in_channel
                    dropped_agents = np.setdiff1d(agents_sending_msg_at_curr_time, agents_sending_msgs_successfully)
                else:
                    agents_sending_msgs_successfully = agents_sending_msg_at_curr_time

                self.dropped_msgs_each_time = dropped_msgs
                for event_agent in agents_sending_msgs_successfully:
                    if self.m_each_agent[event_agent] is None:
                        self.m_each_agent[event_agent] = 0
                        self.tau_m_each_agent[event_agent] = [self.curr_time]
                    else:
                        self.m_each_agent[event_agent] += 1
                        self.tau_m_each_agent[event_agent].append(self.curr_time)
                    self.msg_idx_sent_to_rx_each_agent[event_agent].append(self.m_each_agent[event_agent])
                    self.unack_msg_idx_each_agent[event_agent].append(self.m_each_agent[event_agent])
                    self.tau_m_idx_each_agent[event_agent] += 1
                    self.total_jobs_in_to_rx_channel += 1
                    msg_idx = self.m_each_agent[event_agent]
                    delay = np.random.choice(self.simulated_delays_to_sample_from)
                    tau_m_prime = self.curr_time + delay
                    if self.sorted_tau_m_prime_each_agent[event_agent][0] is None:
                        self.sorted_tau_m_prime_each_agent[event_agent] = [tau_m_prime]
                        self.sorted_tau_m_prime_idx_each_agent[event_agent] = [msg_idx]
                        self.delay_sent_message_each_agent[event_agent][0] = delay
                    else:
                        # sort and re index
                        index = (bisect(self.sorted_tau_m_prime_each_agent[event_agent], tau_m_prime))
                        self.sorted_tau_m_prime_each_agent[event_agent].insert(index, tau_m_prime)
                        self.sorted_tau_m_prime_idx_each_agent[event_agent].insert(index, msg_idx)
                        self.delay_sent_message_each_agent[event_agent].insert(index, delay)
                    self.channel_state_to_rx_each_agent[event_agent][tau_m_prime] = [msg_idx, self.curr_time]
                    self.state_each_agent[event_agent][1] += 1
                    self.time_to_send_each_packet[event_agent] = self.curr_time + np.array(time_till_next_control_each_agent[event_agent])
                self.last_ch_obs_time_each_agent[agents_sending_msg_at_curr_time] = self.total_jobs_in_to_rx_channel
        if self.curr_time > 0:
            self.time_to_send_each_packet[self.time_to_send_each_packet <= self.curr_time] = self.curr_time + 1

        assert (sum(self.state_each_agent[:,1]) == self.total_jobs_in_to_rx_channel)
        assert (sum(self.state_each_agent[:,2]) == self.total_jobs_in_to_sender_channel)
        mean_obs = np.zeros((self.obs_size, 1), dtype=np.float32)
        for o in range(3):
            mean_obs[o] = np.mean(self.state_each_agent[:, o])
            mean_obs[o+3] = np.std(self.state_each_agent[:, o])
        if mean_obs[0] > self.max_x_t_obs:
            mean_obs[0] = self.max_x_t_obs
        reward = self.get_mean_reward(mean_obs, dropped_msgs)
        return reward, mean_obs

    def get_mean_reward(self, mean_obs, dropped_msgs):
        avg_drops = (dropped_msgs / self.number_of_agents)
        reward = - (avg_drops + mean_obs[0][0])     # avg drops + avg aoi
        reward = reward / 100       # to scale down the reward
        return reward

    def get_na_reward(self, dropped_agents):
        reward = np.zeros(self.number_of_agents)
        if dropped_agents is not None:
            reward[dropped_agents] -= 1
        for n in range(self.number_of_agents):
            reward[n] -= self.state_each_agent[n][0]
        reward = reward / 100
        return reward

