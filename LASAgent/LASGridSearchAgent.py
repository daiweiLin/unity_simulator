import csv
import datetime
import os
import time
from collections import deque

import baselines.common.tf_util as U
from baselines import logger
from baselines.ddpg.noise import *
from gym import spaces
from mpi4py import MPI

from LASAgent.InternalEnvironment import InternalEnvironment


class LASGridSearchAgent:
    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity',
                 load_pretrained_agent_flag=False, save_dir=None):
        self.gridsearchagent = GridSearchAgent(agent_name, observation_dim, action_dim, env, env_type, save_dir)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)

    def feed_observation(self, observation):
        """
        Diagram of structure:

        -----------------------------------------------------------------
        |                                             LASBaselineAgent   |
        |                                                                |
        |  action,flag         observation                               |
        |    /\                    |                                     |
        |    |                    \/                                     |
        |  -------------------------------                               |
        |  |    Internal Environment     |                               |
        |  -------------------------------                               |
        |   /\                     |  Flt observation, reward, flag      |
        |   |  action             \/                                     |
        |  ---------------------------                                   |
        |  |      Baseline agent     |                                   |
        |  ---------------------------                                   |
        |                                                                |
        ------------------------------------------------------------------

        """
        take_action_flag = 0

        is_new_observation, filtered_observation, reward = self.internal_env.feed_observation(observation)
        if is_new_observation:
            action, env_reset = self.gridsearchagent.interact(filtered_observation, reward, done=False)
            take_action_flag = 1
            return take_action_flag, action, env_reset
        else:
            return take_action_flag, [], False

    def stop(self):
        self.gridsearchagent.stop()


class GridSearchAgent:

    def __init__(self, agent_name, observation_dim, action_dim, env=None, env_type='Unity', save_dir=None):

        self.name = agent_name
        # =======================================#
        # Get parameters defined in parse_arg() #
        # =======================================#
        args = self.parse_args()
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure()

        # share = False
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)

        # ===================================== #
        # Define observation and action space   #
        # ===================================== #
        if env is None:
            self.env = None
            obs_max = np.array([1.] * observation_dim)
            obs_min = np.array([0.] * observation_dim)
            act_max = np.array([1] * action_dim)
            act_min = np.array([-1] * action_dim)
            self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
            self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)
        else:
            self.env = env
            self.env_type = env_type
            if env_type == 'VREP':
                self.action_space = env.action_space
                self.observation_space = env.observation_space

            elif env_type == 'Unity':
                obs_max = np.array([1.] * observation_dim)
                obs_min = np.array([-1] * observation_dim)
                act_max = np.array([1] * action_dim)
                act_min = np.array([-1] * action_dim)
                self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
                self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)

        self.reward = 0
        self.action = np.zeros(self.action_space.shape[0])
        self.prev_observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # tf.reset_default_graph()

        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()

        assert (np.abs(self.action_space.low) == self.action_space.high).all()  # we assume symmetric actions.

        # Reward histories
        self.episode_rewards_history = deque(maxlen=100)
        self.avg_episode_rewards_history = []

        # =========== #
        # Action grid #
        # =========== #
        self.action_grid = self.get_action_grid(num_samples=args['num_sample_per_para'], para_range=args['para_range'],
                                                search_para=args['search_para'])
        self.search_action_idx = self.get_action_index(args['search_para'])
        self.action_grid_index = 0
        self.epoch_per_comb = args['epochs_per_comb']
        # print(self.action_grid)

        # ===========================#
        # Training cycle parameter #
        # ==========================#

        self.nb_epochs = self.action_grid.shape[0] * self.epoch_per_comb
        self.epoch_cnt = 0
        self.nb_epoch_cycles = args['nb_epoch_cycles']
        self.epoch_cycle_cnt = 0
        self.nb_rollout_steps = args['nb_rollout_steps']
        self.rollout_step_cnt = 0
        self.nb_train_steps = args['nb_train_steps']
        self.training_step_cnt = 0

        # ========================#
        # Model saving           #
        # ========================#
        if save_dir is not None:
            self.model_dir = os.path.join(save_dir, 'model')
            self.log_dir = os.path.join(save_dir, 'log')
        else:
            self.model_dir = os.path.join(os.path.abspath('.'), 'save', 'model')
            self.log_dir = os.path.join(os.path.abspath('.'), 'save', 'log')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # =======================#
        # Initialize tf session #
        # =======================#

        self.sess = U.make_session(num_cpu=1, make_default=True)

        # ==============#
        # logging info #
        # ==============#
        self.episode_reward = 0.
        self.episode_step = 0
        self.episodes = 0
        self.t = 0

        self.start_time = time.time()

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_start_time = time.time()
        self.epoch_actions = []
        self.epoch_episodes = 0

    def interact(self, observation, reward=0, done=False):
        """
        Receive observation and produce action

        """

        # # For the case of simulator only,
        # # since with the simulator, we always use interact() instead of feed_observation()
        env_reset = False
        with self.sess.as_default():
            action = self.generate_PLA_action(self.action_grid[self.action_grid_index])
            # action = self.action_space.sample()

            # Execute next action.

            self.t += 1

            self.episode_reward += reward
            self.episode_step += 1
            self.rollout_step_cnt += 1

            if self.rollout_step_cnt >= self.nb_rollout_steps:
                done = True

            # Book-keeping.
            self.epoch_actions.append(action)
            # Note: self.action correspond to prev_observation
            #       reward correspond to observation
            self._save_log(self.log_dir,
                           [datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), self.prev_observation, self.action,
                            reward])
            self.action = action
            self.reward = reward
            self.prev_observation = observation

            # Logging the training reward info for debug purpose
            if done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.avg_episode_rewards_history.append(self.episode_reward / self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

            # Training
            # At the end of rollout(nb_rollout_steps), it will train the model by nb_train_steps times
            if self.rollout_step_cnt >= self.nb_rollout_steps:
                self.rollout_step_cnt = 0
                self.epoch_cycle_cnt += 1

            # ==========================#
            # Create stats every epoch #
            # ==========================#
            if self.epoch_cycle_cnt >= self.nb_epoch_cycles:
                # rank = MPI.COMM_WORLD.Get_rank()
                mpi_size = MPI.COMM_WORLD.Get_size()
                # Log stats.
                # XXX shouldn't call np.mean on variable length lists
                duration = time.time() - self.start_time
                combined_stats = dict()
                combined_stats['rollout/return'] = np.sum(self.epoch_episode_rewards)
                combined_stats['rollout/return_history'] = np.mean(self.episode_rewards_history)
                combined_stats['rollout/episode_steps'] = np.mean(self.epoch_episode_steps)
                combined_stats['total/duration'] = duration
                combined_stats['total/steps_per_second'] = float(self.t) / float(duration)
                combined_stats['total/episodes'] = self.episodes
                combined_stats['rollout/episodes'] = self.epoch_episodes

                # combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

                def as_scalar(x):
                    if isinstance(x, np.ndarray):
                        assert x.size == 1
                        return x[0]
                    elif np.isscalar(x):
                        return x
                    else:
                        raise ValueError('expected scalar, got %s' % x)

                combined_stats_sums = MPI.COMM_WORLD.allreduce(
                    np.array([as_scalar(x) for x in combined_stats.values()]))
                combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                # Total statistics.
                combined_stats['total/epochs'] = self.epoch_cnt + 1
                combined_stats['total/steps'] = self.t

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')

                self.epoch_cycle_cnt = 0
                self.epoch_cnt += 1
                self.epoch_episode_rewards = []

                if self.epoch_cnt % self.epoch_per_comb == 0:
                    self.action_grid_index += 1
                    env_reset = True

            # ===================#
            # Stop the learning #
            # ===================#

            if self.epoch_cnt >= self.nb_epochs:
                self.stop()

        return action, env_reset

    def _save_log(self, save_dir, data):
        """
        Save action, observation and rewards in a local file
        :param save_dir:
        """
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        file_dir = os.path.join(save_dir, date + ".csv")
        with open(file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)

    def generate_PLA_action(self, grid_action):
        """
        Put test actions into agent's action.
        """

        action = np.zeros(self.action_space.shape)

        for k, a in enumerate(grid_action):
            action[self.search_action_idx[k]] = a
        return action

    def get_action_grid(self, num_samples, para_range, search_para):
        """
        Generate lists of actions to test with.
        :param num_samples: number of samples for each parameter
        :param para_range: value range of each parameter
        :param search_para: names of parameters to be searched
        :return: Dimension: [num_samples**num_para, num_samples]
        """
        num_para = len(search_para)
        actions = np.zeros(shape=(num_samples ** num_para, num_para))

        para = []
        for k, v in enumerate(search_para):
            para.append(np.linspace(para_range[k][0], para_range[k][1], num_samples))

        self.recur_loop(para=para, value=np.array([]), index=np.array([]), matrix=actions)

        return actions

    def recur_loop(self, para, value, index, matrix):
        if len(para) > 1:
            for i in range(len(para[0])):
                new_value = np.append(value, para[0][i])
                new_index = np.append(index, i)
                self.recur_loop(para[1:], new_value, new_index, matrix)
        else:
            num_samples = len(para[0])
            for j in range(len(para[0])):
                row_index = 0
                l = len(index)
                for k in range(l):
                    row_index += index[k] * (num_samples ** (l - k))
                row_index += j
                matrix[int(row_index):] = np.append(value, para[0][j])

    def get_action_index(self, para):
        all_parameters = ['led_ru', 'led_ho', 'led_rd', 'moth_ru', 'moth_ho', 'moth_rd',
                          'I_max', 'ml_gap', 'sma_gap', 'n_gap', 't_sma']
        action_index = []
        for i in range(len(all_parameters)):
            if all_parameters[i] in para:
                action_index.append(i)

        assert len(action_index) == len(para), "Number of action index ({}) does not match number of parameters({}).".format(len(action_index), len(para))
        return action_index

    def parse_args(self):
        """
        This is the place to define training variables. Still using the code from OpenAI Baseline library

        # total step = nb_epochs * nb_epoch_cycles * nb_rollout_steps

        """

        dict_args = dict()
        # dict_args['nb_epochs'] = 1000
        dict_args['nb_epoch_cycles'] = 1
        dict_args['nb_train_steps'] = 5
        dict_args['nb_eval_steps'] = 100
        dict_args['nb_rollout_steps'] = 10

        dict_args['num_sample_per_para'] = 3
        dict_args['para_range'] = [[0, 1]] * 6
        dict_args['search_para'] = ['led_ru', 'led_ho', 'led_rd', 'I_max', 'ml_gap', 'n_gap']
        dict_args['epochs_per_comb'] = 2  # Number of epochs to run for one combination of test parameters

        return dict_args

    def stop(self):
        """
        Stop learning and store the information

        """
        # close the tf session
        self.sess.close()
