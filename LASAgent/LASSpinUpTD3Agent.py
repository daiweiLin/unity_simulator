"""
Created on Sept 16, 2019
Author: Daiwei Lin

Learning agent using TD3 by OpenAI Spinning Up

Adapted from spinup/algos/td3/td3.py
1. Wrap td3() method into TD3Agent class
2. Use Buffer from original td3.py file
"""

import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.algos.td3.td3 import ReplayBuffer
from spinup.utils.run_utils import setup_logger_kwargs

from LASAgent.InternalEnvironment import InternalEnvironment


class LASSpinUpTD3Agent:

    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity',
                 load_pretrained_agent_flag=False, save_dir=None):
        self.spinup_agent = SpinUpTD3Agent(agent_name, observation_dim, action_dim, env, env_type, save_dir)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)

    def feed_observation(self, observation, reward=0, done=False):
        """
        Diagram of structure:

        ------------------------------------------------------------------
        |                                             LASSpinUpPPOAgent  |
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
        |  |      SpinUpPPO agent     |                                  |
        |  ---------------------------                                   |
        |                                                                |
        ------------------------------------------------------------------

        """

        is_new_observation, filtered_observation, reward = self.internal_env.feed_observation(observation)
        if is_new_observation:
            action, reset = self.spinup_agent.interact(filtered_observation, reward, d=done)  # action, reset = self.ppo_agent.interact(filtered_observation, reward, d=done)
            take_action_flag = 1
            return take_action_flag, action, reset
        else:
            take_action_flag = 0
            return take_action_flag, []  # , False


class SpinUpTD3Agent:

    def __init__(self, agent_name, observation_dim, action_dim, env, env_type, save_dir, actor_critic=core.mlp_actor_critic):
        """
       Args:
           env_fn : A function which creates a copy of the environment.
               The environment must satisfy the OpenAI Gym API.
           actor_critic: A function which takes in placeholder symbols
               for state, ``x_ph``, and action, ``a_ph``, and returns the main
               outputs from the agent's Tensorflow computation graph:
               ===========  ================  ======================================
               Symbol       Shape             Description
               ===========  ================  ======================================
               ``pi``       (batch, act_dim)  | Deterministically computes actions
                                              | from policy given states.
               ``q1``       (batch,)          | Gives one estimate of Q* for
                                              | states in ``x_ph`` and actions in
                                              | ``a_ph``.
               ``q2``       (batch,)          | Gives another estimate of Q* for
                                              | states in ``x_ph`` and actions in
                                              | ``a_ph``.
               ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                              | ``pi`` for states in ``x_ph``:
                                              | q1(x, pi(x)).
               ===========  ================  ======================================
           ac_kwargs (dict): Any kwargs appropriate for the actor_critic
               function you provided to TD3.
           seed (int): Seed for random number generators.
           steps_per_epoch (int): Number of steps of interaction (state-action pairs)
               for the agent and the environment in each epoch.
           epochs (int): Number of epochs to run and train agent.
           replay_size (int): Maximum length of replay buffer.
           gamma (float): Discount factor. (Always between 0 and 1.)
           polyak (float): Interpolation factor in polyak averaging for target
               networks. Target networks are updated towards main networks
               according to:
               .. math:: \\theta_{\\text{targ}} \\leftarrow
                   \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
               where :math:`\\rho` is polyak. (Always between 0 and 1, usually
               close to 1.)
           pi_lr (float): Learning rate for policy.
           q_lr (float): Learning rate for Q-networks.
           batch_size (int): Minibatch size for SGD.
           start_steps (int): Number of steps for uniform-random action selection,
               before running real policy. Helps exploration.
           act_noise (float): Stddev for Gaussian exploration noise added to
               policy at training time. (At test time, no noise is added.)
           target_noise (float): Stddev for smoothing noise added to target
               policy.
           noise_clip (float): Limit for absolute value of target policy
               smoothing noise.
           policy_delay (int): Policy will only be updated once every
               policy_delay times for each update of the Q-networks.
           max_ep_len (int): Maximum length of trajectory / episode / rollout.
           logger_kwargs (dict): Keyword args for EpochLogger.
           save_freq (int): How often (in terms of gap between epochs) to save
               the current policy and value function.
       """
        #============#
        # Parse args #
        #============#
        args = self.parse_args()
        # mpi_fork(args['cpu'])  # run parallel code with mpi

        tf.set_random_seed(int(time.time()))
        np.random.seed(int(time.time()))
        # tf.set_random_seed(seed)
        # np.random.seed(seed)

        self.seed = args['seed'] = 0  # Discard as this will cause identical results for PLA
        # self.seed = args['exp_name'] = 'td3'
        self.save_freq = args['save_freq'] = 1

        self.epochs = args['epochs']
        self.steps_per_epoch = args['steps_per_epoch']
        self.replay_size = args['replay_size']
        self.polyak = args['polyak']
        self.pi_lr = args['pi_lr']
        self.q_lr = args['q_lr']
        self.batch_size = args['batch_size']
        self.start_steps = args['start_steps']
        self.act_noise = args['act_noise']
        self.target_noise = args['target_noise']
        self.noise_clip = args['noise_clip']
        self.policy_delay = args['policy_delay']
        self.max_ep_len = args['max_ep_len']
        self.gamma = args['gamma']

        self.ac_kwargs = args['ac_kwargs']
        self.logger = EpochLogger(**args['logger_kwargs'])
        # Disabled because this will cause trouble when interacting with Unity.
        # For its function, see https://spinningup.openai.com/en/latest/utils/logger.html#spinup.utils.logx.Logger.save_config
        self.logger.save_config(locals())


        self.env, self.test_env = env, env
        if env_type == "Unity":
            obs_max = np.array([1.] * observation_dim)
            obs_min = np.array([-1] * observation_dim)
            act_max = np.array([1] * action_dim)
            act_min = np.array([-1] * action_dim)

            self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
            self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)
            obs_dim = self.observation_space.shape[0]
            act_dim = self.action_space.shape[0]
        else:
            # Gym environment
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            self.action_space = env.action_space
            self.observation_space = env.observation_space

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = env.action_space.high[0]

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = actor_critic(self.x_ph, self.a_ph, **self.ac_kwargs)

        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _ = actor_critic(self.x2_ph, self.a_ph, **self.ac_kwargs)

        # Target Q networks
        with tf.variable_scope('target', reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
            self.a2 = pi_targ + epsilon
            self.a2 = tf.clip_by_value(self.a2, -self.act_limit, self.act_limit)

            # Target Q-values, using action from target policy
            _, self.q1_targ, self.q2_targ, _ = actor_critic(self.x2_ph, self.a2, **self.ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)
        backup = tf.stop_gradient(self.r_ph + self.gamma * (1 - self.d_ph) * min_q_targ)

        # TD3 losses
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        q1_loss = tf.reduce_mean((self.q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((self.q2 - backup) ** 2)
        self.q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))
        self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph, 'a': self.a_ph}, outputs={'pi': self.pi, 'q1': self.q1, 'q2': self.q2})

        self.start_time = time.time()
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        self.total_steps = self.steps_per_epoch * self.epochs

        # Counters and returns for main loop in interact()
        self.step_cnt = 0
        self.ep_len = 0
        self.ep_ret = 0
        self.a = self.action_space.sample()
        self.o = self.observation_space.sample()

    def parse_args(self):
        """
        This is the place to define training variables. Still using the code from OpenAI Baseline library

        # total step = nb_epochs * nb_epoch_cycles * nb_rollout_steps

        """

        dict_args = dict()
        dict_args['hid'] = 300 # size of each hidden layer
        dict_args['l'] = 1 # number of layers
        dict_args['seed'] = 0  # Discard as this will cause identical results for PLA
        dict_args['exp_name'] = 'td3'
        dict_args['save_freq'] = 1

        dict_args['epochs'] = 100  # default 100
        dict_args['steps_per_epoch'] = 5000 # default 5000
        dict_args['replay_size'] = int(1e6)
        dict_args['polyak'] = 0.995
        dict_args['pi_lr'] = 1e-3
        dict_args['q_lr'] = 1e-3
        dict_args['batch_size'] = 100
        dict_args['start_steps'] = 10000
        dict_args['act_noise'] = 0.1
        dict_args['target_noise'] = 0.2
        dict_args['noise_clip'] = 0.5
        dict_args['policy_delay'] = 2
        dict_args['max_ep_len'] = 25  # default 1000

        dict_args['gamma'] = 0.99

        dict_args['ac_kwargs'] = dict(hidden_sizes=[dict_args['hid']] * dict_args['l'])
        dict_args['logger_kwargs'] = setup_logger_kwargs(dict_args['exp_name'], dict_args['seed'])
        return dict_args

    def get_action(self, o, noise_scale):
        a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(self.action_space.shape[0])
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self, n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = self.get_action(o, 0)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def interact(self, o, r, d):
        # Main loop: collect experience in env and update/log each epoch
        env_reset = False
        if self.step_cnt > self.start_steps:
            a = self.get_action(o, self.act_noise)
        else:
            a = self.env.action_space.sample()

        self.step_cnt += 1

        self.ep_ret += r
        self.ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if self.ep_len == self.max_ep_len else d

        # Store experience to replay buffer
        self.replay_buffer.store(self.o, self.a, r, o, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o
        self.a = a

        if d or (self.ep_len == self.max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            for j in range(self.ep_len):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                feed_dict = {self.x_ph: batch['obs1'],
                             self.x2_ph: batch['obs2'],
                             self.a_ph: batch['acts'],
                             self.r_ph: batch['rews'],
                             self.d_ph: batch['done']
                             }
                q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
                outs = self.sess.run(q_step_ops, feed_dict)
                self.logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % self.policy_delay == 0:
                    # Delayed policy update
                    outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
                    self.logger.store(LossPi=outs[0])

            self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
            self.ep_ret, self.ep_len = 0, 0
            env_reset = True
            # o, r, d = self.env.reset(), 0, False  # <=======================================================change later

        # End of epoch wrap-up
        if self.step_cnt > 0 and self.step_cnt % self.steps_per_epoch == 0:
            epoch = self.step_cnt // self.steps_per_epoch

            # # Save model
            # if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
            #     self.logger.save_state({'env': self.env}, None)

            # Test the performance of the deterministic version of the agent.
            self.test_agent()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('TestEpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('TestEpLen', average_only=True)
            self.logger.log_tabular('TotalEnvInteracts', self.step_cnt)
            self.logger.log_tabular('Q1Vals', with_min_and_max=True)
            self.logger.log_tabular('Q2Vals', with_min_and_max=True)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossQ', average_only=True)
            self.logger.log_tabular('Time', time.time() - self.start_time)
            self.logger.dump_tabular()

        if self.step_cnt > self.total_steps:
            self.stop()

        return a, env_reset

    def stop(self):
        """
        Stop learning and store the information

        """
        # close the tf session
        self.sess.close()
