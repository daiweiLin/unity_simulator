import argparse
import time
import datetime
import os
import logging
import pickle
import csv
from collections import deque

from gym import spaces
import tensorflow as tf
from mpi4py import MPI
import numpy as np

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *




class LASRandomAgent:
    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity', load_pretrained_agent_flag=False, save_dir=None):
        self.randomAgent = RandomAgent(agent_name, observation_dim, action_dim, env, env_type, save_dir)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)

    def feed_observation(self,observation):
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
            action = self.randomAgent.interact(filtered_observation, reward, done=False)
            take_action_flag = 1
            return take_action_flag, action
        else:
            return take_action_flag, []

    def stop(self):
        self.randomAgent.stop()


class InternalEnvironment:
    def __init__(self,observation_dim, action_dim, num_observation):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_observation = num_observation

        self.observation_cnt = 0
        self.observation_group = np.zeros((num_observation, observation_dim))

    def feed_observation(self, observation):
        """
        1. Feed observation into internal environment
        2. perform filtering
        3. calculate reward
        :param observation:
        :return:
        """

        flt_observation = np.zeros((1,self.observation_dim), dtype=np.float32)
        reward = 0
        # stack observations
        self.observation_group[self.observation_cnt] = observation
        self.observation_cnt += 1

        # Apply filter once observation group is fully updated
        # After that, calculate the reward based on filtered observation
        if self.observation_cnt >= self.num_observation:
            self.observation_cnt = 0
            # self.flt_prev_observation = self.flt_observation
            flt_observation = self._filter(self.observation_group)
            is_new_observation = 1

            reward = self._cal_reward(flt_observation)

        else:
            is_new_observation = 0

        return is_new_observation, flt_observation, reward

    def take_action(self,action):
        take_action_flag = True
        return take_action_flag, action

    def _cal_reward(self, flt_observation):
        """
        Calculate the extrinsic rewards based on the filtered observation
        Filtered observation should have same size as observation space
        :return: reward
        """
        reward = 0
        for i in range(flt_observation.shape[0]):
            reward += flt_observation[i]
        return reward

    def _filter(self, signal):
        """
        Averaging filter

        signal: numpy matrix, one row is one observation

        """
        return np.mean(signal, axis = 0)


class RandomAgent:
    def __init__(self, agent_name, observation_dim, action_dim, env=None, env_type='VREP', save_dir=None):

        self.name = agent_name
        #=======================================#
        # Get parameters defined in parse_arg() #
        #=======================================#
        args = self.parse_args()
        noise_type = args['noise_type']
        layer_norm = args['layer_norm']
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
        # max_action = self.action_space.high
        # logger.info('scaling actions by {} before executing in env'.format(max_action))


        # Reward histories

        self.episode_rewards_history = deque(maxlen=100)
        self.avg_episode_rewards_history = []

        #===========================#
        # Training cycle parameter #
        #==========================#

        self.nb_epochs = args['nb_epochs']
        self.epoch_cnt = 0
        self.nb_epoch_cycles = args['nb_epoch_cycles']
        self.epoch_cycle_cnt = 0
        self.nb_rollout_steps = args['nb_rollout_steps']
        self.rollout_step_cnt = 0
        self.nb_train_steps = args['nb_train_steps']
        self.training_step_cnt = 0

        #========================#
        # Model saving           #
        #========================#
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


        #=======================#
        # Initialize tf session #
        #=======================#

        self.sess = U.make_session(num_cpu=1, make_default=True)

        # self.saver = tf.train.Saver()

        #==============#
        # logging info #
        #==============#
        self.episode_reward = 0.
        self.episode_step = 0
        self.episodes = 0
        self.t = 0

        # epoch = 0
        self.start_time = time.time()

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_episode_eval_rewards = []
        self.epoch_episode_eval_steps = []
        self.epoch_start_time = time.time()
        self.epoch_actions = []
        self.epoch_qs = [] # Q values
        self.epoch_episodes = 0
        self.param_noise_adaption_interval = 50



    def interact(self, observation, reward = 0, done = False):
        """
        Receive observation and produce action

        """

        # # For the case of simulator only,
        # # since with the simulator, we always use interact() instead of feed_observation()
        # if self.env is not None:
        #     self.observe(observation)
        with self.sess.as_default():
            action = self.action_space.sample()
            # assert action.shape == self.action_space.shape

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
            # if self.action is not None:
            #     self.agent.store_transition(self.prev_observation, self.action, reward, observation, done)

            self._save_log(self.log_dir,
                           [datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), self.prev_observation, self.action, reward])
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

                # self.agent.reset() # <<<<???? not sure
                # For simulation on unity, no reset needed. Because the simulation is to simulate ROM experiment.
                # if self.env is not None:
                #     obs = self.env.reset()


            # Training
            # At the end of rollout(nb_rollout_steps), it will train the model by nb_train_steps times
            if self.rollout_step_cnt >= self.nb_rollout_steps:

                self.rollout_step_cnt = 0
                self.epoch_cycle_cnt += 1

            #==========================#
            # Create stats every epoch #
            #==========================#
            if self.epoch_cycle_cnt >= self.nb_epoch_cycles:
                # rank = MPI.COMM_WORLD.Get_rank()
                mpi_size = MPI.COMM_WORLD.Get_size()
                # Log stats.
                # XXX shouldn't call np.mean on variable length lists
                duration = time.time() - self.start_time
                # stats = self.agent.get_stats()
                combined_stats = dict()
                combined_stats['rollout/return'] = np.sum(self.epoch_episode_rewards)
                combined_stats['rollout/return_history'] = np.mean(self.episode_rewards_history)
                combined_stats['rollout/episode_steps'] = np.mean(self.epoch_episode_steps)
                combined_stats['total/duration'] = duration
                combined_stats['total/steps_per_second'] = float(self.t) / float(duration)
                combined_stats['total/episodes'] = self.episodes
                combined_stats['rollout/episodes'] = self.epoch_episodes
                combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)

                def as_scalar(x):
                    if isinstance(x, np.ndarray):
                        assert x.size == 1
                        return x[0]
                    elif np.isscalar(x):
                        return x
                    else:
                        raise ValueError('expected scalar, got %s' % x)

                combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
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
            #===================#
            # Stop the learning #
            #===================#

            if self.epoch_cnt >= self.nb_epochs:
                self.stop()

        return action


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

    def parse_args(self):
        """
        This is the place to define training variables. Still using the code from OpenAI Baseline library

        # total step = nb_epochs * nb_epoch_cycles * nb_rollout_steps

        """

        dict_args = dict()
        dict_args['render_eval'] = False
        dict_args['layer_norm'] = True
        dict_args['render'] = False
        dict_args['normalize_returns'] = False
        dict_args['normalize_observations'] = False
        dict_args['critic_l2_reg'] = 1e-2
        dict_args['batch_size'] = 64
        dict_args['actor_lr'] = 1e-4
        dict_args['critic_lr'] = 1e-3
        dict_args['popart'] = False
        dict_args['gamma'] = 0.99
        dict_args['reward_scale'] = 1.
        dict_args['clip_norm'] = None
        dict_args['nb_epochs'] = 2000
        dict_args['nb_epoch_cycles'] = 2
        dict_args['nb_train_steps'] = 20
        dict_args['nb_eval_steps'] = 100
        dict_args['nb_rollout_steps'] = 50
        dict_args['noise_type'] = 'adaptive-param_0.2'
        dict_args['evaluation'] = False

        return dict_args

    def stop(self):
        """
        Stop learning and store the information

        """
        # if using the simulator
        if self.env is not None:
            if self.env_type == "V-REP":
                print("close connection to V-REP simulator")
                self.env.close_connection()
            else:
                print("close Unity.")
                self.env.close()

        # close the tf session
        self.sess.close()


