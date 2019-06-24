"""
Created on 2019-02-05 11:58 AM
Author: Daiwei Lin

Reference:
    1. https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb
    2. https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md
"""

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment
from Prescripted_behaviour_timing import *
from LASAgent.LASBaselineAgent import *
from Visitor_behaviour import *


def LAS_behavior(p, action_dimension):
    if np.random.rand(1) < p:
        return np.random.randn(action_dimension)
    else:
        return np.zeros(action_dimension)


def init(mode, num_visitors, unity_dir, no_graphics=False, interact_with_app=True, save_dir=None):

    env_name = unity_dir
    # Start the environment
    #    interact_with_app == True: interact with application
    #    interact_with_app == False: interact with Unity scene starting by click play in Unity

    if interact_with_app == True:
        env = UnityEnvironment(file_name=env_name, seed=1, no_graphics=no_graphics)
    else:
        env = UnityEnvironment(file_name=None, seed=1)

    # # Set the default brain to work with
    # default_brain = env.brain_names[0]

    # # Reset the environment
    # env_info = env.reset(train_mode=train_mode)[default_brain]
    #
    # # Examine the state space for the default brain
    # print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))
    #
    # # Examine the observation space for the default brain
    # for observation in env_info.visual_observations:
    #     print("Agent observations look like:")
    #     if observation.shape[3] == 3:
    #         plt.imshow(observation[0, :, :, :])
    #     else:
    #         plt.imshow(observation[0, :, :, 0])

    ############
    # Visitors #
    ############

    visitor_bh = Visitor_behaviour(num_visitors, 24)

    ##################
    # Learning Agent #
    ##################
    if save_dir is not None:
        summary_path = os.path.join(save_dir, 'summary')
    else:
        summary_path = os.path.join(os.path.abspath('.'), 'save', 'summary')

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    os.environ['OPENAI_LOGDIR'] = summary_path
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'

    if mode == 'PLA':
        # initialize pre-scripted behaviour class
        ROM_sculpture = Sculpture(node_num=24, sma_num=6, led_num=1, moth_num=1)
        behaviour = Behaviour(ROM_sculpture, system_freq=10)

        # initialize ml agent
        agent = LASBaselineAgent('Baseline_Agent', observation_dim=24, action_dim=11, num_observation=10,
                                 env=env, env_type='Unity', load_pretrained_agent_flag=False, save_dir=save_dir)
        return env, visitor_bh, agent, behaviour

    elif mode == 'SARA':
        # initialize ml agent
        agent = LASBaselineAgent('Baseline_Agent', observation_dim=24, action_dim=168, num_observation=10,
                                 env=env, env_type='Unity', load_pretrained_agent_flag=False, save_dir=save_dir)
        return env, visitor_bh, agent, None


def run(mode, behaviour, agent, visitors_behaviour):

    if visitors_behaviour.num_visitors > 1:
        visitor_brain_name = 'GroupVisitorBrain'
    else:
        visitor_brain_name = 'VisitorBrain'
    LAS_brain_name = 'LASBrain'

    print("Learning:")
    env_info = env.reset(train_mode=train_mode)
    done = False
    reward = 0
    observation = env_info[LAS_brain_name].vector_observations[0]
    simulator_time = observation[-1]
    observation = observation[0:-1]
    take_action_flag = 1  # switch for debugging

    if mode == 'PLA':

        total_steps = agent.baseline_agent.nb_epochs * agent.baseline_agent.nb_epoch_cycles * \
                      agent.baseline_agent.nb_rollout_steps
        s = 0
        while s <= total_steps - 1:

            # para_action = agent.interact(observation, reward, done)
            take_para_action_flag, para_action = agent.feed_observation(observation)
            if take_para_action_flag:
                behaviour.set_parameter(para_action)
                s += 1
                # print('s={}'.format(s))

            LAS_action = behaviour.step(observation, simulator_time)
            LAS_action = LAS_action + [take_action_flag]
            # print("LAS Action:{}".format(LAS_action))

            # action = agent.interact(observation, reward, done)
            # take_action_flag = 1
            # LAS_action = action.tolist() + [take_action_flag]

            Visitor_action = visitors_behaviour.step(env_info[visitor_brain_name].vector_observations[0])

            action = {LAS_brain_name: LAS_action, visitor_brain_name: Visitor_action}
            # print("LED_Action: {}".format(LAS_action[:24]))
            env_info = env.step(action)

            reward = env_info[LAS_brain_name].rewards[0]
            observation = env_info[LAS_brain_name].vector_observations[0]
            # <use time in simulator>
            simulator_time = observation[-1]
            observation = observation[0:-1]
            # <end of use time in simulator>
            done = env_info[LAS_brain_name].local_done[0]

    elif mode == 'SARA':

        total_steps = agent.baseline_agent.nb_epochs * agent.baseline_agent.nb_epoch_cycles * \
                      agent.baseline_agent.nb_rollout_steps
        LAS_action = agent.baseline_agent.action_space.sample().tolist() + [take_action_flag]
        s = 0
        while s <= total_steps - 1:

            take_SARA_action_flag, new_LAS_action = agent.feed_observation(observation)
            if take_SARA_action_flag:
                LAS_action = new_LAS_action.tolist() + [take_action_flag]
                s += 1

            Visitor_action = visitors_behaviour.step(env_info[visitor_brain_name].vector_observations[0])

            action = {LAS_brain_name: LAS_action, visitor_brain_name: Visitor_action}
            # print("LED_Action: {}".format(LAS_action[:24]))
            env_info = env.step(action)

            reward = env_info[LAS_brain_name].rewards[0]
            observation = env_info[LAS_brain_name].vector_observations[0]
            # <use time in simulator>
            simulator_time = observation[-1]
            observation = observation[0:-1]
            # <end of use time in simulator>
            done = env_info[LAS_brain_name].local_done[0]




if __name__ == '__main__':

    train_mode = True  # Whether to run the environment in training or inference mode
    learning_mode = 'PLA'  # 'SARA', 'PLA'
    n_visitors = 5

    is_sharcnet = False
    if len(sys.argv) > 1:
        is_sharcnet = sys.argv[1] == "True"
        learning_mode = sys.argv[2]
        n_visitors = int(sys.argv[3])

    if is_sharcnet:
        interact_with_app = True
        no_graphics = True
        if n_visitors > 1:
            unity_dir = 'unity_executable/multi_visitor/LAS_Simulator'
        else:
            unity_dir = 'unity_executable/single_visitor/LAS_Simulator'
    else:
        interact_with_app = False
        no_graphics = False
        unity_dir = 'LAS-Scenes/Unity/LAS_Simulator'

    date = datetime.datetime.today().strftime('%Y-%m-%d-%H%M%S')
    save_dir = os.path.join(os.path.abspath('.'), 'save', learning_mode, date)

    print("Training Case Parameters:")
    print("Is_sharcnet={}, training_mode={}, learning_mode={}, number_of_visitors={}, interact_with_app={}".format(is_sharcnet, train_mode, learning_mode, n_visitors, interact_with_app))

    env, visitors_bh, agent, bh = init(mode=learning_mode, num_visitors=n_visitors,
                                       unity_dir=unity_dir, no_graphics=no_graphics,
                                       interact_with_app=interact_with_app,
                                       save_dir=save_dir)
    run(mode=learning_mode, behaviour=bh, agent=agent, visitors_behaviour=visitors_bh)
