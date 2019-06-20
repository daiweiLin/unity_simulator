"""
Created on 2019-06-06 
Author: Daiwei Lin

Reference:
    1. https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb
    2. https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md
"""

# 1. Load dependencies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment

def visitor_behavior(observation, node_number):
    # print("visitor observation: {}".format(observation))
    visitor_action = np.random.uniform(low=-1, high=1, size=2)*np.array([12.5, 7.5])
    x = []
    y = []
    for i in range(node_number):
        if observation[i] > 0:
            x.append(observation[node_number+i*2])
            y.append(observation[node_number+i*2+1])
    if len(x) > 1:
        random = np.random.randint(low=0, high=len(x)-1)
        visitor_action = [x[random], y[random]]
        # print("visitor find light at {}".format(visitor_action))
    elif len(x) == 1:
        visitor_action = [x[0], y[0]]
        # print("visitor find light at {}".format(visitor_action))

    return visitor_action


def LAS_behavior(p, action_dimension):
    if np.random.rand(1) < p:
        return np.random.randn(action_dimension)
    else:
        return np.zeros(action_dimension)


if __name__ == '__main__':
    # Detect Operating System and Choose the Unity environment binary to launch
    # if sys.platform == "linux" or sys.platform == "linux2":
    #     env_name = os.path.join('..', 'LAS_Simulator_Linux', 'LAS_Simulator')
    # elif sys.platform == "win32":
    #     env_name = os.path.join('..', 'LAS_Simulator_Windows', 'LAS_Simulator')
    # elif sys.platform == "darwin":
    #     env_name = os.path.join('..', 'LAS_Simulator_Mac', 'LAS_Simulator')


    train_mode = True  # Whether to run the environment in training or inference mode
    env_name = 'LAS-Scenes/Unity/LAS_Simulator'
    # 3. Start the environment
    #    interact_with_app == True: interact with application
    #    interact_with_app == False: interact with Unity scene starting by click play in Unity
    interact_with_app = False
    if interact_with_app == True:
        env = UnityEnvironment(file_name=env_name, seed=1, no_graphics=False)
    else:
        env = UnityEnvironment(file_name=None, seed=1)

    # Set the default brain to work with
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    # import pdb
    # pdb.set_trace()
    # 4. Examine the observation and state spaces
    # Reset the environment
    env_info = env.reset(train_mode=train_mode)[default_brain]

    # Examine the state space for the default brain
    print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

    # Examine the observation space for the default brain
    for observation in env_info.visual_observations:
        print("Agent observations look like:")
        if observation.shape[3] == 3:
            plt.imshow(observation[0,:,:,:])
        else:
            plt.imshow(observation[0,:,:,0])



    env_info = env.reset(train_mode=train_mode)
    done = False
    reward = 0
    observation = env_info['LASBrain'].vector_observations[0]

    take_action_flag = 1 # switch for debugging

    action_size = env.brains['LASBrain'].vector_action_space_size
    s = 0
    while s <= 1000:
        env_info = env.reset(train_mode=train_mode)

        LAS_action = np.random.randn(action_size[0] - 1).tolist()
        LAS_action = LAS_action + [take_action_flag]
        # print("LAS Action:{}".format(LAS_action))

        Visitor_action = visitor_behavior(env_info['VisitorBrain'].vector_observations[0], node_number=24)

        action = {'LASBrain': LAS_action, 'VisitorBrain': Visitor_action}
        # print("LED_Action: {}".format(LAS_action[:24]))
        env_info = env.step(action)

        reward = env_info['LASBrain'].rewards[0]
        observation = env_info['LASBrain'].vector_observations[0]
        done = env_info['LASBrain'].local_done[0]

        if reward > 0:
            print("s={},reward={}".format(s,reward))

        s += 1