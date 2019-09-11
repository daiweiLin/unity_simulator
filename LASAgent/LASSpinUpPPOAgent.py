"""
Learning agent using PPO by OpenAI Spinning Up

"""
import time
import datetime
import os

from gym import spaces
import tensorflow as tf
from mpi4py import MPI
import numpy as np

from spinup.algos.ppo.ppo import PPOBuffer
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.run_utils import setup_logger_kwargs
# from LASAgent.LASBaselineAgent import InternalEnvironment


# class LASSpinUpPPOAgent:
#
#     def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity',
#                  load_pretrained_agent_flag=False, save_dir=None):
#         self.ppo_agent = SpinUpPPOAgent(env)
#         self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)
#
#     def feed_observation(self,observation):
#         """
#         Diagram of structure:
#
#         -----------------------------------------------------------------
#         |                                             LASSpinUpPPOAgent  |
#         |                                                                |
#         |  action,flag         observation                               |
#         |    /\                    |                                     |
#         |    |                    \/                                     |
#         |  -------------------------------                               |
#         |  |    Internal Environment     |                               |
#         |  -------------------------------                               |
#         |   /\                     |  Flt observation, reward, flag      |
#         |   |  action             \/                                     |
#         |  ---------------------------                                   |
#         |  |      SpinUpPPO agent     |                                  |
#         |  ---------------------------                                   |
#         |                                                                |
#         ------------------------------------------------------------------
#
#         """
#         take_action_flag = 0
#
#         is_new_observation, filtered_observation, reward = self.internal_env.feed_observation(observation)
#         if is_new_observation:
#             action = self.ppo_agent.interact(filtered_observation, reward, d=False)
#             take_action_flag = 1
#             return take_action_flag, action
#         else:
#             return take_action_flag, []
#
#     # def stop(self):
#     #     self.ppo_agent.stop()


class SpinUpPPOAgent:

    """
    This is adapted from ppo() method from spinup.algos.ppo.ppo

    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    """

    def __init__(self, env, actor_critic=core.mlp_actor_critic):
        """
        Args:
            env : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: A function which takes in placeholder symbols
                for state, ``x_ph``, and action, ``a_ph``, and returns the main
                outputs from the agent's Tensorflow computation graph:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       (batch, act_dim)  | Samples actions from policy given
                                               | states.
                ``logp``     (batch,)          | Gives log probability, according to
                                               | the policy, of taking actions ``a_ph``
                                               | in states ``x_ph``.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                               | the policy, of the action sampled by
                                               | ``pi``.
                ``v``        (batch,)          | Gives the value estimate for states
                                               | in ``x_ph``. (Critical: make sure
                                               | to flatten this!)
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the actor_critic
                function you provided to PPO.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.
            gamma (float): Discount factor. (Always between 0 and 1.)
            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while
                still profiting (improving the objective function)? The new policy
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.)
            pi_lr (float): Learning rate for policy optimizer.
            vf_lr (float): Learning rate for value function optimizer.
            train_pi_iters (int): Maximum number of gradient descent steps to take
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)
            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.
            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used
                for early stopping. (Usually small, 0.01 or 0.05.)
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        # ============ #
        #  Parameters  #
        # ============ #
        args = self.parse_args()
        # mpi_fork(args['cpu'])  # run parallel code with mpi

        logger_kwargs = setup_logger_kwargs(args['exp_name'], args['seed'])
        ac_kwargs = args['ac_kwargs']

        self.epochs = args['epochs']
        self.steps_per_epoch = args['steps_per_epoch']
        self.pi_lr = args['pi_lr']
        self.vf_lr = args['vf_lr']
        self.train_pi_iters = args['train_pi_iters']
        self.train_v_iters = args['train_v_iters']
        self.max_ep_len = args['max_ep_len']
        self.target_kl = args['target_kl']
        self.clip_ratio = args['clip_ratio'] = 0.2
        self.lam = args['lam'] = 0.97
        self.gamma = args['gamma'] = 0.99
        self.save_freq = args['save_freq']

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.seed = args['seed'] + 10000 * proc_id()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        self.env = env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = self.env.action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(self.env.observation_space, self.env.action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, self.gamma, self.lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # PPO objectives
        ratio = tf.exp(self.logp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(self.adv_ph > 0, (1 + self.clip_ratio) * self.adv_ph, (1 - self.clip_ratio) * self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)  # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Sync params across processes
        self.sess.run(sync_all_params())

        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph}, outputs={'pi': self.pi, 'v': self.v})


        #===============================#
        # Initialize iteration counters #
        #===============================#
        self.epoch_cnt = 0
        self.step_cnt = 0
        self.ep_ret = 0
        self.ep_len = 0
        # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        self.start_time = time.time()

    def parse_args(self):
        """
        This is the place to define training variables. Still using the code from OpenAI Baseline library

        # total step = nb_epochs * nb_epoch_cycles * nb_rollout_steps

        """

        dict_args = dict()
        dict_args['hid'] = 64 # size of each hidden layer
        dict_args['l'] = 2 # number of layers

        dict_args['seed'] = 0
        dict_args['cpu'] = 4 # MPI
        dict_args['exp_name'] = 'ppo'

        dict_args['epochs'] = 50
        dict_args['steps_per_epoch'] = 4000
        dict_args['pi_lr'] = 3e-4
        dict_args['vf_lr'] = 1e-3
        dict_args['train_pi_iters'] = 80
        dict_args['train_v_iters'] = 80
        dict_args['max_ep_len'] = 1000
        dict_args['target_kl'] = 0.01
        dict_args['clip_ratio'] = 0.2
        dict_args['lam'] = 0.97
        dict_args['gamma'] = 0.99
        dict_args['save_freq'] = 10
        dict_args['ac_kwargs'] = dict(hidden_sizes=[dict_args['hid']]*dict_args['l'])
        return dict_args

    def interact(self, o, r, d):
        """
        Receive observation and produce action

        """
        env_reset = False



        a, v_t, logp_t = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: o.reshape(1, -1)})

        # save and log
        self.buf.store(o, a, r, v_t, logp_t)
        self.logger.store(VVals=v_t)

        # o, r, d, _ = self.env.step(a[0])
        self.ep_ret += r
        self.ep_len += 1

        terminal = d or (self.ep_len == self.max_ep_len)
        if terminal or (self.step_cnt == self.local_steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % self.ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = r if d else self.sess.run(self.v, feed_dict={self.x_ph: o.reshape(1, -1)})
            self.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
            env_reset =True
            self.ep_ret = 0
            self.ep_len = 0
            # o, r, d, self.ep_ret, self.ep_len = self.env.reset(), 0, False, 0, 0

        self.step_cnt += 1
        if self.step_cnt >= self.local_steps_per_epoch:
            # Perform PPO update!
            self.update()

            # Log info about epoch
            self.logger.log_tabular('Epoch', self.epoch_cnt)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (self.epoch_cnt + 1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - self.start_time)
            self.logger.dump_tabular()

            self.step_cnt = 0
            self.epoch_cnt += 1

        if self.epoch_cnt >= self.epochs:
            self.stop()

        return a[0], env_reset

    def update(self):
        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        # Training
        for i in range(self.train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        self.logger.store(StopIter=i)
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=inputs)
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    def stop(self):
        """
        Stop learning and store the information

        """
        # close the tf session
        self.sess.close()
