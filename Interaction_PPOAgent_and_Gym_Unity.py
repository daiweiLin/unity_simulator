"""
Created on 2019-09-11 01:21 PM
Author: Daiwei Lin

Main body copied from Test_Simulator.py
Only tested with SARA mode
"""

import sys
import os
import numpy as np
import tensorflow as tf
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityWorkerInUseException
from Prescripted_behaviour_timing import *
from LASAgent.LASSpinUpPPOAgent import *
from Visitor_behaviour import *
# from LASAgent.LASRandomAgent import *
import gym


def LAS_behavior(p, action_dimension):
    if np.random.rand(1) < p:
        return np.random.randn(action_dimension)
    else:
        return np.zeros(action_dimension)


def init(mode, num_visitors, unity_dir, save_dir, no_graphics=False, interact_with_app=True):
    """
    :param mode: Random, SARA or PLA
    :param num_visitors: 1 or 5
    :param unity_dir:
    :param no_graphics: True if running jobs on SHARCNET
    :param interact_with_app:
        True: interact with application;
        False: interact with Unity scene starting by click play in Unity
    :param save_dir: log file directory
    :return:
    """
    env_name = unity_dir if interact_with_app else None

    # Initialized unity environment. Each environment requires an unique worker_id.
    worker_id = 0
    while worker_id < 10:
        try:
            env = UnityEnvironment(file_name=env_name, seed=1, no_graphics=no_graphics, worker_id=worker_id)
        except UnityWorkerInUseException:
            print("Worker ID {} is in use.".format(worker_id))
            worker_id += 1
        except Exception as e:
            print("Other exceptions:{}".format(e))
        else:
            print("UnityEnvironment initialized with worker_id={}".format(worker_id))
            break
    #==== for Gym env =============#
    # env = gym.make("CartPole-v1")
    # =============================#
    save_dir = save_dir + "-" + str(worker_id)

    ############
    # Visitors #
    ############
    visitor_bh = Visitor_behaviour(num_visitors, 2)

    ##################
    # Learning Agent #
    ##################
    summary_path = os.path.join(save_dir, 'summary')

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    os.environ['OPENAI_LOGDIR'] = summary_path
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'

    if mode == 'PLA':
        # initialize pre-scripted behaviour class
        ROM_sculpture = Sculpture(node_num=24, sma_num=6, led_num=1, moth_num=1)
        behaviour = Behaviour(ROM_sculpture, system_freq=10)

        # initialize ml agent
        agent = SpinUpPPOAgent(env=env)
        return env, visitor_bh, agent, behaviour

    elif mode == 'SARA':
        # initialize ml agent
        agent = LASSpinUpPPOAgent('SpinUP PPO Agent', observation_dim=24, action_dim=168, num_observation=1,
                                 env=env, env_type='Unity', load_pretrained_agent_flag=False, save_dir=save_dir)
        return env, visitor_bh, agent, None
    # elif mode == 'Random':
    #     # initialize random action agent
    #     agent = LASRandomAgent('RandomAgent', observation_dim=24, action_dim=168, num_observation=10, env=env,
    #                            env_type='Unity', save_dir=save_dir)
    #     return env, visitor_bh, agent, None


def run(mode, behaviour, agent, visitors_behaviour):

    if visitors_behaviour.num_visitors > 1:
        visitor_brain_name = 'GroupVisitorBrain'
    else:
        visitor_brain_name = 'VisitorBrain'
    LAS_brain_name = 'LASBrain'

    print("Learning:")
    env_info = env.reset(train_mode=train_mode)

    coordinates = env_info[visitor_brain_name].vector_observations[0][24:72]
    visitors_behaviour.setup(coordinates)

    observation = env_info[LAS_brain_name].vector_observations[0]
    done = False
    reward = 0
    simulator_time = observation[-1]
    observation = observation[0:-1]
    take_action_flag = 1  # switch for debugging

    #========== For Gym environment only ========================#
    # observation, reward, done, info = env.reset(), 0, False, ""
    #============================================================#
    if mode == 'PLA':
        print("mode in PLA")

    elif mode == 'SARA':

        total_steps = agent.ppo_agent.local_steps_per_epoch * agent.ppo_agent.epochs
        LAS_action = agent.ppo_agent.action_space.sample().tolist() + [take_action_flag]
        s = 0
        while s <= total_steps - 1:
            #=============[ For Unity environment only ]============================================#
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
            # simulator_time = observation[-1]
            observation = observation[0:-1]
            # <end of use time in simulator>
            done = env_info[LAS_brain_name].local_done[0]
            #=========== End of code for Unity Environment ======================================#

            #==========[ For Gym environment Only ]===========================================================#
            # agent.feed_observation(observation, reward, done)
            # if take_action_flag:
            #
            # # Visitor_action = visitors_behaviour.step(env_info[visitor_brain_name].vector_observations[0])
            # observation, reward, done, info = env.step(action)
            # if reset:
            #     observation, reward, done, info = env.reset(), 0, False, ""
            #========== End of Code for Gym Environment ====================================================#

    elif mode == 'Random':
        print("mode in Random")


    # print("close Unity.")
    env.close()

if __name__ == '__main__':

    train_mode = True  # Whether to run the environment in training or inference mode
    learning_mode = 'SARA'  # 'SARA', 'PLA', 'Random'
    n_visitors = 1

    is_sharcnet = False
    job_id = ""
    if len(sys.argv) > 1:
        is_sharcnet = sys.argv[1] == "True"
        learning_mode = sys.argv[2]
        n_visitors = int(sys.argv[3])
        job_id = sys.argv[4]

    if is_sharcnet:
        interact_with_app = True
        no_graphics = True
        if n_visitors > 1:
            unity_dir = 'unity_executable/multi_visitor/LAS_Simulator'
        else:
            unity_dir = 'unity_executable/single_visitor/LAS_Simulator'
    else:
        interact_with_app = True
        no_graphics = False
        unity_dir = 'LAS-Scenes/Unity/LAS_Simulator'

    date = datetime.datetime.today().strftime('%Y-%m-%d-%H%M%S')
    save_dir = os.path.join(os.path.abspath('.'), 'save', learning_mode, date+"-"+job_id)

    print("Training Case Parameters:")
    print("Is_sharcnet={}, training_mode={}, learning_mode={}, number_of_visitors={}, interact_with_app={}".format(is_sharcnet, train_mode, learning_mode, n_visitors, interact_with_app))

    env, visitors_bh, agent, bh = init(mode=learning_mode, num_visitors=n_visitors,
                                       unity_dir=unity_dir, no_graphics=no_graphics,
                                       interact_with_app=interact_with_app,
                                       save_dir=save_dir)
    run(mode=learning_mode, behaviour=bh, agent=agent, visitors_behaviour=visitors_bh)