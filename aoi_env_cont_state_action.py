import gym
import numpy as np
from gym import spaces
from bisect import bisect
import portion as P

class AOI_env(gym.Env):
    def __init__(self, simulation_timesteps=30, number_particles=100, number_of_agents=2, config='pomfc', results_dir=None,
                 action_space_size=2, weighted_action=False, use_true_state=False, use_belief_state=False,
                 lambda_val=1, channel_known=False, use_particles=False, eval=False,
                 drops=False, r=None, true_state_thr=None):
        if r is not None:
            self.r = float(r)
        if true_state_thr is not None:
            self.true_state_thr = true_state_thr
        self.drops = drops
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
        if self.config == 'pomfc':
            self.action_space = spaces.MultiDiscrete([2] * self.action_space_size)
            self.observation_space = spaces.Tuple((spaces.Box(low=0, high=obs_high, shape=(self.obs_size,), dtype=np.float32),
                                    spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)))     # +2 for \nu
        elif self.config in ('na_dec', 'rnd', 'a1', 'thr'):
            self.action_space = spaces.MultiDiscrete([2] * self.action_space_size)
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
                           spaces.Box(low=1, high=self.max_time, shape=(self.action_space_size,), dtype=np.float32)))
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
        self.state_each_agent = np.zeros([self.number_of_agents, 4], dtype=np.float32)
        self.total_jobs_in_to_sender_channel = 0
        self.total_jobs_in_to_rx_channel = 0
        self.all_events = []
        self.all_event_agents = []
        self.time_each_agent = [[0] for n in range(self.number_of_agents)]
        self.last_aoi_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.ack_info_each_agent = [{} for n in range(self.number_of_agents)]
        # for each particle simulate the delays it can experience, using the same parameters as the original channel delay
        self.simulated_delays_each_particle_each_agent = []
        self.simulated_delays_to_sample_from = np.random.exponential(self.lambda_val, size=1000) + 1
        self.simulated_delays_to_sample_from[self.simulated_delays_to_sample_from > 3] = 3
        for i in range(self.number_of_agents):
            self.simulated_delays_each_particle_each_agent.append(self.simulated_delays_to_sample_from)
        self.infos = {}
        self.done = False
        self.curr_time = 0
        self.prev_time = 0
        self.time_to_send_each_packet = np.random.uniform(0, self.max_initial_timeslot, size=self.number_of_agents)
        self.delay_to_rx_each_agent_each_job = [[] for n in range(self.number_of_agents)]
        self.channel_state_obs_each_agent = [self.total_jobs_in_to_rx_channel] * self.number_of_agents
        self.mean_channel_state = self.total_jobs_in_to_rx_channel
        self.last_ch_obs_time_each_agent = np.zeros(self.number_of_agents, dtype=np.float32)
        self.expected_belief_x_t_each_agent = [0] * self.number_of_agents
        self.std_belief_x_t_each_agent = [0] * self.number_of_agents
        self.curr_particles_each_agent = [[0] * self.number_of_particles] * self.number_of_agents
        self.nu = np.array([1.0, 0.0], dtype=np.float32)
        if self.config == 'pomfc':
            empty_obs = np.zeros((self.obs_size), dtype=np.float32)
            curr_obs = [empty_obs, self.nu]
        elif self.config in ('na_dec', 'rnd', 'a1', 'thr'):
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
        self.division1 = P.IntervalDict()
        self.division1[P.closed(0, 1)] = 0
        self.division1[P.closed(1, 2)] = 1
        self.division1[P.closed(2, 3)] = 2
        self.division1[P.closed(3, 4)] = 3
        self.division1[P.closed(4, 5)] = 4
        self.division1[P.closed(5, 6)] = 5
        self.division1[P.closed(6, 7)] = 6
        self.division1[P.closed(7, 8)] = 7
        self.division1[P.closed(8, 9)] = 8
        self.division1[P.closed(9, 10)] = 9
        self.division1[P.closed(10, 11)] = 10
        self.division1[P.closed(11, 12)] = 11
        self.division1[P.closed(12, 13)] = 12
        self.division1[P.closed(13, 14)] = 13
        self.division1[P.closed(14, 15)] = 14
        self.division1[P.closed(15, 100)] = 15
        self.last_when_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.last_rcvd_obs_each_agent = [0 for n in range(self.number_of_agents)]
        self.last_aoi_obs_time_each_agent = [0 for n in range(self.number_of_agents)]
        self.expected_belief_over_time_each_agent = [[] for n in range(self.number_of_agents)]
        self.std_belief_over_time_each_agent = [[] for n in range(self.number_of_agents)]
        return curr_obs

    def step(self, action_list):
        if self.config in ('rnd', 'a1', 'thr'):
            action_each_agent = action_list[0]
            time_till_next_contol_each_agent = action_list[1]
        else:
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

        # assert sum(self.nu) <= 1
        # if self.use_belief_state:
        self.expected_belief_x_t_each_agent = []
        self.std_belief_x_t_each_agent = []
        self.curr_particles_each_agent = []
        if self.config not in ('rnd', 'a1'):
            if not self.use_true_state:
                for n in range(self.number_of_agents):
                    particles = self.all_particles_each_agent[n]
                    expected_belief, std_belief = self.calculate_belief_from_particles(particles, self.curr_time)
                    self.expected_belief_x_t_each_agent.append(expected_belief)
                    self.std_belief_x_t_each_agent.append(std_belief)
                    self.expected_belief_over_time_each_agent[n].append(expected_belief)
                    self.std_belief_over_time_each_agent[n].append(std_belief)
        if self.config == 'pomfc':
            if self.use_true_state:
                nu_1 = self.total_jobs_in_to_rx_channel / self.max_to_rx_channel_capacity
                self.nu = np.array([1 - nu_1, nu_1], dtype=np.float32)
                curr_obs = [mean_obs, self.nu]
            else:
                mean_belief = np.mean(self.expected_belief_x_t_each_agent)
                std_belief = np.std(self.expected_belief_x_t_each_agent)
                mean_obs[0] = mean_belief
                mean_obs[3] = std_belief
                # you assume that you still know how many are in each channel from each agent
                nu_1 = self.mean_channel_state / self.max_to_rx_channel_capacity
                self.nu = np.array([1 - nu_1, nu_1], dtype=np.float32)
                curr_obs = [mean_obs, self.nu]

        elif self.config in ('na_dec', 'rnd', 'a1', 'thr'):
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
                    obs[:, 1] = self.std_belief_x_t_each_agent  # std over belief over aoi
            obs[:, -1] = np.sum(self.state_each_agent[:, 1:3], axis=1)      # total unacked jobs
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

        # # plotting
        if self.curr_time >= self.episode_timesteps -1:
            chosen_idx = []
            i = 0
            while True:
                id = np.random.choice(self.number_of_agents)
                if self.obs_times_each_agent[id][0] is not None:
                    if id not in chosen_idx:
                        chosen_idx.append(id)
                i += 1
                if len(chosen_idx) == 5 or i == 100:
                    break

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
        ratio = np.sum(self.state_each_agent[:, 1:3], axis=1) / (
                sum(np.sum(self.state_each_agent[:, 1:3], axis=1)) + 1e-10)
        if self.weighted_action:    # radial basis function
            for n in range(self.number_of_agents):
                r = self.mid_space - self.expected_belief_x_t_each_agent[n]     # use belief state to extract
                r_squared = -np.square(r)
                expon_weights = np.exp(r_squared/(2 * (self.sigma**2)))
                normalised_weights = expon_weights/sum(expon_weights)
                if self.config in ('na_dec'):
                    weighted_actions = normalised_weights * action_list[n]
                else:
                    weighted_actions = normalised_weights * action_list
                if len(weighted_actions) == self.action_space_size:
                    action_each_agent.append(np.round(sum(weighted_actions)))
                    time_till_next_control_each_agent.append(1)
                else:
                    action_each_agent.append(np.round(sum(weighted_actions[0])))
                    time_till_next_control_each_agent.append(sum(weighted_actions[1]))

        else:
            # always use belief aoi for action extraction
            for n in range(self.number_of_agents):
                if self.use_true_state:
                    all_idx.append(self.division1[self.x_t_each_agent[n][-1]])
                else:
                    all_idx.append(self.division1[self.expected_belief_x_t_each_agent[n]])

            if len(action_list) == self.action_space_size or len(action_list[0]) == self.action_space_size:
                if self.config in ('pomfc'):
                    action_each_agent = action_list[all_idx]
                elif self.config in ('na_dec', 'rnd', 'a1', 'thr'):
                    for n in range(self.number_of_agents):
                        action_each_agent.append(action_list[n][all_idx[n]])
                elif self.config == 'na_cen':
                    for n in range(self.number_of_agents):
                        action_each_agent.append(action_list[all_idx[n]])
                else:
                    y_t_each_agent = self.state_each_agent[:, 3]
                    all_idx = []
                    for n in range(self.number_of_agents):
                        all_idx.append(self.division1[y_t_each_agent[n]])
                    action_each_agent = action_list[all_idx]
                time_till_next_control_each_agent = np.ones(self.number_of_agents)

            else:
                if self.config in ('pomfc'):
                    time_till_next_control_each_agent = action_list[1][all_idx]
                    action_each_agent = action_list[0][all_idx]
                elif self.config in ('na_dec', 'rnd', 'a1', 'thr'):
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
                        all_idx.append(self.division1[y_t_each_agent[n]])
                    time_till_next_control_each_agent = action_list[1][all_idx]
                    action_each_agent = action_list[0][all_idx]
        time_till_next_control_each_agent = np.array(time_till_next_control_each_agent)
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
        all_msg_rcvng_times =  []
        channel_to_rx_state_at_start = self.total_jobs_in_to_rx_channel
        rx_rcvd_msg = []
        agents_rcvd_acks = []
        if self.curr_time != 0:  # update x_t and particles based on acks or msg rcvd or not
            for n in range(self.number_of_agents):
                if self.sorted_obs_times_each_agent[n][0] is not None:
                    if len(np.where((np.array(self.sorted_obs_times_each_agent[n]) <= self.curr_time) & (np.array(self.sorted_obs_times_each_agent[n]) > self.prev_time))[0])> 0:
                        agents_rcvd_acks.append(n)
                if self.sorted_tau_m_prime_each_agent[n][0] is not None:
                    if len(np.where((np.array(self.sorted_tau_m_prime_each_agent[n]) <= self.curr_time) & (np.array(self.sorted_tau_m_prime_each_agent[n]) > self.prev_time))[0]) > 0:
                        rx_rcvd_msg.append(n)

            # for agents not receiving any obs at this time, should be assigned to 0
            for n in range(self.number_of_agents):
                if n not in agents_rcvd_acks:
                    self.state_each_agent[n][3] = 0
            if rx_rcvd_msg:  # update true x_t of agnets and particle update as before
                for event_agent in rx_rcvd_msg:
                    # check how many are observed in this timeslot, could be more than 1
                    arr_times = (np.array(self.sorted_tau_m_prime_each_agent[event_agent]))
                    total_msgs_arriving = len(np.where((arr_times <= self.curr_time) & (arr_times > self.prev_time))[0])
                    last_msg_rcvng_time = self.prev_time
                    for a in range(total_msgs_arriving):
                        rcvd_msg_idx = self.sorted_tau_m_prime_idx_each_agent[event_agent][self.tau_m_prime_idx_each_agent[event_agent]]
                        msg_rcvng_time = self.sorted_tau_m_prime_each_agent[event_agent][self.tau_m_prime_idx_each_agent[event_agent]]
                        all_msg_rcvng_times.append(msg_rcvng_time)
                        time_past = msg_rcvng_time - last_msg_rcvng_time
                        if len(self.msg_idx_rcvd_by_rx_each_agent[event_agent]) == 0 or \
                                sum(rcvd_msg_idx > np.array(self.msg_idx_rcvd_by_rx_each_agent[event_agent])) == \
                                len(self.msg_idx_rcvd_by_rx_each_agent[event_agent]):
                            self.msg_idx_rcvd_by_rx_each_agent[event_agent].append(rcvd_msg_idx)
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.time_each_agent[event_agent].append(msg_rcvng_time)
                            self.x_t_each_agent[event_agent].append(self.delay_sent_message_each_agent[event_agent][rcvd_msg_idx])
                            self.time_each_agent[event_agent].append(msg_rcvng_time)
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
                            self.ack_info_each_agent[event_agent][eta_prime] = [msg_rcvng_time, self.x_t_each_agent[event_agent][-1]]
                            if a == (total_msgs_arriving - 1):
                                time_left = self.curr_time - msg_rcvng_time
                                self.time_each_agent[event_agent].append(self.curr_time)
                                self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_left)
                            self.total_jobs_in_to_sender_channel += 1
                            self.state_each_agent[event_agent][2] += 1
                        else:
                            if a == (total_msgs_arriving - 1):
                                time_past = self.curr_time - last_msg_rcvng_time
                                self.time_each_agent[event_agent].append(self.curr_time)
                            else:
                                time_past = msg_rcvng_time - last_msg_rcvng_time
                                self.time_each_agent[event_agent].append(msg_rcvng_time)
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.obsolete_msg_idx_each_agent[event_agent].append(rcvd_msg_idx)
                        self.tau_m_prime_idx_each_agent[event_agent] += 1
                        self.total_jobs_in_to_rx_channel -= 1
                        self.state_each_agent[event_agent][1] -= 1
                        last_msg_rcvng_time = msg_rcvng_time
            if agents_rcvd_acks:
                for event_agent in agents_rcvd_acks:
                    # check how many are observed in this timeslot, could be more than 1
                    arr_times = (np.array(self.sorted_obs_times_each_agent[event_agent]))
                    total_msgs_arriving = len(np.where((arr_times <= self.curr_time) & (arr_times > self.prev_time))[0])
                    for _ in range(total_msgs_arriving):
                        self.msg_idx_rcvd_by_sender_each_agent[event_agent].append(
                            self.obs_times_idx_each_agent[event_agent])
                        obs_time = self.obs_times_each_agent[event_agent][self.obs_times_idx_each_agent[event_agent]]
                        rho = self.ack_info_each_agent[event_agent][obs_time][0]
                        x_rho = self.ack_info_each_agent[event_agent][obs_time][1]  # rcvd observation
                        self.last_when_obs_time_each_agent[event_agent] = obs_time
                        self.last_rcvd_obs_each_agent[event_agent] = x_rho
                        self.last_aoi_obs_time_each_agent[event_agent] = rho
                        self.state_each_agent[event_agent][3] = x_rho
                        prev_time = np.array([rho] * self.number_of_particles).astype(float)
                        x_rho_all = np.array([x_rho] * self.number_of_particles).astype(float)
                        self.unack_msg_idx_each_agent[event_agent].remove(self.obs_times_idx_each_agent[event_agent])
                        virtual_particles = [[[rho], [x_rho]] for _ in range(self.number_of_particles)]
                        tot_unack_msgs_particles_n = len(self.unack_msg_idx_each_agent[event_agent])
                        if self.config not in ('rnd', 'a1'):
                            if not self.use_true_state:
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
                        # del self.channel_state_to_sender_each_agent[event_agent][self.curr_time]
                        self.state_each_agent[event_agent][2] -= 1
                        if event_agent not in rx_rcvd_msg:
                            time_past = self.curr_time - self.prev_time
                            self.x_t_each_agent[event_agent].append(self.x_t_each_agent[event_agent][-1] + time_past)
                            self.time_each_agent[event_agent].append(self.curr_time)
        channel_to_rx_state_at_end = self.total_jobs_in_to_rx_channel
        if self.curr_time > 0:
            for n in range(self.number_of_agents):
                if n not in rx_rcvd_msg and n not in agents_rcvd_acks:  # if 1 and 2 event did not occur
                    time_past = self.curr_time - self.prev_time
                    self.x_t_each_agent[n].append(self.x_t_each_agent[n][-1] + time_past)
                    self.time_each_agent[n].append(self.curr_time)
                self.state_each_agent[n, 0] = self.x_t_each_agent[n][-1]
        assert(channel_to_rx_state_at_end <= channel_to_rx_state_at_start)
        dropped_msgs = 0
        agents_sending_msgs_successfully = []
        if self.curr_time > 0:
            if self.config not in ('rnd', 'a1'):
                # if self.config == 'thr' and self.true_state_thr == 'f':
                if not self.use_true_state:
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
            agents_should_msg_at_curr_time = np.where((np.array(self.time_to_send_each_packet) <= self.curr_time) & (np.array(self.time_to_send_each_packet) > self.prev_time))[0]
            agents_sending_msg_at_curr_time = np.intersect1d(msg_sending_agents_at_curr_time, agents_should_msg_at_curr_time)
            if len(agents_sending_msg_at_curr_time) > 0:  # event == 0
                initial_remaining_slots_in_channel = self.max_to_rx_channel_capacity - channel_to_rx_state_at_start
                total_arr_jobs = len(agents_sending_msg_at_curr_time)
                if total_arr_jobs > initial_remaining_slots_in_channel:
                    # choose agents randomly to drop jobs
                    if channel_to_rx_state_at_end == channel_to_rx_state_at_start:  # no new channels got free
                        agents_sending_msgs_successfully = agents_sending_msg_at_curr_time[
                            np.argsort(self.time_to_send_each_packet[agents_sending_msg_at_curr_time])][
                        0:initial_remaining_slots_in_channel]
                        dropped_msgs = total_arr_jobs - initial_remaining_slots_in_channel
                    else:
                        #check when channel got free and if any agent sent after that
                        sorted_times_channel_got_free = np.sort(all_msg_rcvng_times)
                        sorted_sending_agents_send_msg = agents_sending_msg_at_curr_time[
                            np.argsort(self.time_to_send_each_packet[agents_sending_msg_at_curr_time])]
                        sorted_times_agent_send_msg = self.time_to_send_each_packet[sorted_sending_agents_send_msg]
                        time_initial_free_slots = np.ones(initial_remaining_slots_in_channel) * self.prev_time
                        total_free_slot_times = np.concatenate((time_initial_free_slots, sorted_times_channel_got_free))
                        agents_sending_msgs_successfully = []
                        remaining_slots_in_channel = 0
                        for t in total_free_slot_times:
                            if len(sorted_times_agent_send_msg) > 0:
                                if sum(sorted_times_agent_send_msg > t) > 0:
                                    remaining_slots_in_channel += 1
                                which_agent_can_send = sorted_sending_agents_send_msg[np.where(sorted_times_agent_send_msg > t)[0]]
                                if len(which_agent_can_send) > 0:
                                    if which_agent_can_send[0] not in agents_sending_msgs_successfully:
                                        agents_sending_msgs_successfully.append(which_agent_can_send[0])
                                    sorted_sending_agents_send_msg = sorted_sending_agents_send_msg[1:]
                                    sorted_times_agent_send_msg = sorted_times_agent_send_msg[1:]

                        agents_sending_msgs_successfully = np.array(agents_sending_msgs_successfully, dtype=int)
                        dropped_msgs = np.max([(total_arr_jobs - remaining_slots_in_channel), 0])
                else:
                    agents_sending_msgs_successfully = agents_sending_msg_at_curr_time
                assert (dropped_msgs >= 0)
                self.dropped_msgs_each_time = dropped_msgs
                for event_agent in agents_sending_msgs_successfully:
                    if self.m_each_agent[event_agent] is None:
                        self.m_each_agent[event_agent] = 0
                        self.tau_m_each_agent[event_agent] = [self.time_to_send_each_packet[event_agent]]
                    else:
                        self.m_each_agent[event_agent] += 1
                        self.tau_m_each_agent[event_agent].append(self.time_to_send_each_packet[event_agent])
                    self.msg_idx_sent_to_rx_each_agent[event_agent].append(self.m_each_agent[event_agent])
                    self.unack_msg_idx_each_agent[event_agent].append(self.m_each_agent[event_agent])
                    self.tau_m_idx_each_agent[event_agent] += 1
                    self.total_jobs_in_to_rx_channel += 1
                    msg_idx = self.m_each_agent[event_agent]
                    delay = np.random.choice(self.simulated_delays_to_sample_from)
                    tau_m_prime = self.time_to_send_each_packet[event_agent] + delay
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
                    self.state_each_agent[event_agent][1] += 1
                self.last_ch_obs_time_each_agent[agents_sending_msg_at_curr_time] = self.total_jobs_in_to_rx_channel

        if self.curr_time > 0:
            if len(agents_sending_msgs_successfully) > 0:
                self.time_to_send_each_packet[agents_sending_msgs_successfully] += time_till_next_control_each_agent[agents_sending_msgs_successfully]
            idx = np.where(self.time_to_send_each_packet <= self.curr_time)[0]
            if len(idx) > 0:
                for n in idx:
                    self.time_to_send_each_packet[n] = (self.curr_time + np.random.rand(1))

        assert (sum(self.state_each_agent[:,1]) == self.total_jobs_in_to_rx_channel)
        assert (sum(self.state_each_agent[:,2]) == self.total_jobs_in_to_sender_channel)
        mean_obs = np.zeros((self.obs_size), dtype=np.float32)
        for o in range(3):
            mean_obs[o] = np.mean(self.state_each_agent[:, o])
            mean_obs[o+3] = np.std(self.state_each_agent[:, o])
        if mean_obs[0] > self.max_x_t_obs:
            mean_obs[0] = self.max_x_t_obs
        reward = self.get_mean_reward(mean_obs, dropped_msgs)
        return reward, mean_obs

    def get_mean_reward(self, mean_obs, dropped_msgs):
        if self.drops:
            avg_drops = (dropped_msgs / self.number_of_agents)
        else:
            avg_drops = 0
        reward = - (avg_drops + mean_obs[0])     # avg drops + avg aoi
        reward = reward / 100       # to scale down the reward
        return reward

    def get_na_reward(self, dropped_agents, dropped_msgs):
        reward = -dropped_msgs
        reward -= sum(self.state_each_agent[:, 0])
        reward = reward / self.number_of_agents
        return reward

