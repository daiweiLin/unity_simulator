import numpy as np

class InternalEnvironment:
    def __init__(self,observation_dim, action_dim, num_observation):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_observation = num_observation

        self.observation_cnt = 0
        self.observation_group = np.zeros((num_observation, observation_dim))

        # for the case of augmented observation
        if observation_dim > 24:
            self.reward_observation = 24
        else:
            self.reward_observation = observation_dim

    def feed_observation(self, observation):
        """
        1. Feed observation into internal environment
        2. perform filtering
        3. calculate reward
        :param observation:
        :return:
        """

        flt_observation = np.zeros((1, self.observation_dim), dtype=np.float32)
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
        for i in range(self.reward_observation):
            reward += flt_observation[i]
        return reward

    def _filter(self, signal):
        """
        Averaging filter

        signal: numpy matrix, one row is one observation

        """
        return np.mean(signal, axis = 0)
