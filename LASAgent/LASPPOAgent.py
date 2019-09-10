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
from LASAgent.LASBaselineAgent import InternalEnvironment


class LASSpinUpPPOAgent:

    def __init__(self, agent_name, observation_dim, action_dim, num_observation=20, env=None, env_type='Unity',
                 load_pretrained_agent_flag=False, save_dir=None):
        self.ppo_agent = SpinUpPPOAgent(agent_name, observation_dim, action_dim, env, env_type,
                                        load_pretrained_agent_flag, save_dir)
        self.internal_env = InternalEnvironment(observation_dim, action_dim, num_observation)

    def feed_observation(self,observation):
        """
        Diagram of structure:

        -----------------------------------------------------------------
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
        take_action_flag = 0

        is_new_observation, filtered_observation, reward = self.internal_env.feed_observation(observation)
        if is_new_observation:
            action = self.ppo_agent.interact(filtered_observation, reward, done=False)
            take_action_flag = 1
            return take_action_flag, action
        else:
            return take_action_flag, []

    # def stop(self):
    #     self.ppo_agent.stop()


class SpinUpPPOAgent:

    """
    This is adapted from ppo() method from spinup.algos.ppo.ppo

    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    """

    def __init__(self, agent_name, observation_dim, action_dim, env=None, env_type='VREP',
                 load_pretrained_agent_flag=False, save_dir=None):
        self.agent_name = agent_name

    def ppo(self, env, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
            steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
            target_kl=0.01, logger_kwargs=dict(), save_freq=10):
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

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        seed += 10000 * proc_id()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        env = env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space

        # Inputs to computation graph
        x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

        # Every step, get: action, value, and logprob
        get_action_ops = [pi, v, logp_pi]

        # Experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # PPO objectives
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning)
        approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
        approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
        train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Sync params across processes
        sess.run(sync_all_params())

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

        def update():
            inputs = {k: v for k, v in zip(all_phs, buf.get())}
            pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

            # Training
            for i in range(train_pi_iters):
                _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
                kl = mpi_avg(kl)
                if kl > 1.5 * target_kl:
                    logger.log('Early stopping at step %d due to reaching max kl.' % i)
                    break
            logger.store(StopIter=i)
            for _ in range(train_v_iters):
                sess.run(train_v, feed_dict=inputs)

            # Log changes from update
            pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
            logger.store(LossPi=pi_l_old, LossV=v_l_old,
                         KL=kl, Entropy=ent, ClipFrac=cf,
                         DeltaLossPi=(pi_l_new - pi_l_old),
                         DeltaLossV=(v_l_new - v_l_old))

        start_time = time.time()
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})

                # save and log
                buf.store(o, a, r, v_t, logp_t)
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a[0])
                ep_ret += r
                ep_len += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal or (t == local_steps_per_epoch - 1):
                    if not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                    buf.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Perform PPO update!
            update()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()