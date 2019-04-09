#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 18 2019

@author: daiwei.lin
"""


from LASAgent.LASBaselineAgent import *
import sys
from collections import deque
from mlagents.envs import UnityEnvironment


def initialize_unit_env(train_mode=True):

    """
    :param train_mode: True if to run the environment in training, false if in inference mode
    :return:
    """
    # Instantiate environment object

    env_name = "unity_executable/navigation/navigation"  # Name of the Unity environment binary to launch

    print("Python version:")
    print(sys.version)

    # check Python version
    if sys.version_info[0] < 3:
        raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

    env = UnityEnvironment(file_name=None)

    # Set the visitor brain to work with
    default_brain = env.brain_names[0]
    # brain_name = "VisitorBrain"
    brain = env.brains[default_brain]

    # # Reset the environment
    # env_info = env.reset(train_mode=train_mode)[default_brain]

    # # Examine the state space for the default brain
    # print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

    return env, default_brain, brain



if __name__ == '__main__':

    train_mode = True
    unity_env, default_brain, brain = initialize_unit_env(train_mode)

    # env_obs_convert = np.array([1 / 3.15, 1 / 3.15, 1 / 3.15, 1 / 4, 1 / 4, 1 / 4, 1 / 10, 1 / 10])

    env_info = unity_env.reset(train_mode=train_mode)[default_brain]
    done = False
    reward = 0
    print(len(env_info.vector_observations))
    observation = env_info.vector_observations[0]
    episode_rewards = deque(10*[0], 10)
    episode_r = 0
    episode = 0

    destination = np.array([[-4, -5], [-4, 5], [4, 5], [4, -5]])
    i = 0

    print("Start:")
    while True:

        # env_info = unity_env.step(destination[i, :])[default_brain ]
        env_info = unity_env.step({default_brain: destination[i, :], "Visitor_two_Brain": [0, 0]})
        # print("take action")

        # reward = env_info.rewards[0]
        # done = env_info.local_done[0]

        # if it has multiple brains, identify which brain. The env_info is returned as dictionary object.
        observation = env_info[default_brain].vector_observations[0]

        if observation[-1] == 0:
            print("Reach destination {}".format(destination[i,:]))
            i += 1
            if i >= 4:
                i = 0

    unity_env.close()

