import numpy as np


class Visitor_behaviour:

    def __init__(self, num_visitors, attractions):
        self.num_visitors = num_visitors
        # self.visitor_stay_time = stay_time # seconds

        self.attractions = attractions
        self.node_number = None
        self.dist_matrix = None

        # self.visitor_start_ts = np.zeros(self.num_visitors, dtype=np.float64)
        self.visitor_prev_dest = [None]*self.num_visitors
        self.visitor_arrived_at_node = [False]*self.num_visitors

    def setup(self, coordinates):
        self.node_number = int(len(coordinates) / 2)
        self.dist_matrix = self._cal_distance(coordinates)

    def _cal_distance(self, coordinates):
        num_nodes = int(len(coordinates)/2)
        dis_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)

        for i in range(num_nodes):
            start_point = np.array([coordinates[i*2], coordinates[i*2+1]])
            for j in range(i, num_nodes):
                end_point = np.array([coordinates[j*2], coordinates[j*2+1]])
                dis_matrix[i, j] = np.linalg.norm(end_point-start_point)
                dis_matrix[j, i] = dis_matrix[i, j] # because of symmetry
        # print(dis_matrix)
        return dis_matrix

    def find_hot_spot(self, observation, attraction):
        """
        Find the spot with most activities in the area
        The score is calculated using the sum of inverse of distances times observation values.
        :return: index of selected spot if exists, return None o.w.
        """
        if attraction == "SMA":
            # Attract to SMA
            # Pre-process SMA observations in groups of 6
            observation_sma = observation[self.node_number:]
            observation_attracting = []
            for i in range(self.node_number):
                observation_attracting.append(np.sum(observation_sma[6*i:6*i+6]))
        else:
            # Attract to LED
            observation_attracting = observation[:self.node_number]

        observation_attracting = np.array(observation_attracting)
        heat = np.zeros(self.node_number)
        for i in range(self.node_number):
            distance = self.dist_matrix[i, :]
            distance[i] = 1 # change this to avoid numerical error
            heat[i] = np.sum(observation_attracting * np.reciprocal(distance))

        # print("heat={}".format(heat))                         #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Change here
        if np.sum(heat > 0) > 0:
            return heat.argmax()
        else:
            return None

    def step(self, observation):
        """

        :param observation: [LED1, LED2, ... LEDn,
                            SMA1-1, SMA1-2,...,SMA1-6, SMA2-1, ..., SMAn-1,...SMAn-6,
                            x1, y1, x2, y2, ... xn, yn,
                            is_arrived_1, is_arrived_2, ... is_arrived_m,
                            ingame_time] 3*n + m + 1 elements in total.
                            n = number of nodes;
                            m = number of visitors;
        :return:  [visitor1.x, visitor1.y, visitor2.x, visitor2.y, ... visitorN.x, visitorN.y]
        """
        visitor_actions = list()

        obs = observation[0: 7 * self.node_number]
        positions = observation[7 * self.node_number: 9 * self.node_number]
        is_arrv = observation[9 * self.node_number: -1]
        # ingame_t = observation[-1]  # this parameter is only used in pre-scripted_behaviour
        for v in range(self.num_visitors):

            # duration = 0
            dest = None

            if is_arrv[v]:
                prev_dest = self.visitor_prev_dest[v]
                if prev_dest is None:
                    # Visitor just arrived at a random position, so he/she moves to next location
                    dest = self.find_hot_spot(observation=obs, attraction=self.attractions[v])
                    self.visitor_prev_dest[v] = dest
                else:
                    # Visitor just arrived at a node, so he/she wants to stay as long as the light is ON
                    if obs[prev_dest] <= 0:
                        dest = self.find_hot_spot(observation=obs, attraction=self.attractions[v])
                        self.visitor_prev_dest[v] = dest
                    else:
                        dest = self.visitor_prev_dest[v]

            else:
                dest = self.find_hot_spot(observation=obs, attraction=self.attractions[v])

            if dest is not None:
                # Found a destination

                # print("Destination = Node{}".format(dest))
                visitor_actions.append(positions[2 * dest])
                visitor_actions.append(positions[2 * dest + 1])
            else:
                # Random select a position in space
                random_dest = np.random.uniform(low=-1, high=1, size=2) * np.array([10.0, 6.5])
                visitor_actions = visitor_actions + random_dest.tolist()
                # print("Destination = Random {}".format(random_dest))

        # # Find turned-on LEDs and randomly choose
        # x = []
        # y = []
        # for i in range(self.node_number):
        #     if observation[i] > 0:
        #         x.append(observation[self.node_number + i * 2])
        #         y.append(observation[self.node_number + i * 2 + 1])
        #
        # if len(x) > 1:
        #     visitor_actions = []
        #     for i in range(self.num_visitors):
        #         random = np.random.randint(low=0, high=len(x) - 1)
        #         visitor_actions.append(x[random])
        #         visitor_actions.append(y[random])
        #         # print("visitor {} find light at {}".format(i, visitor_actions))
        # elif len(x) == 1:
        #     visitor_actions = [x[0], y[0]] * self.num_visitors
        #     # print("visitor find light at {}".format(visitor_action))
        # else:
        #     visitor_actions = np.random.uniform(low=-1, high=1, size=self.num_visitors * 2) * np.array(
        #         [10.0, 6.5] * self.num_visitors)

        return visitor_actions


if __name__ == "__main__":
    # This section is for testing Visitor_behaviour Class

    visitor_bh = Visitor_behaviour(num_visitors=1, attraction=["SMA"])
    print(visitor_bh.dist_matrix)
    print(visitor_bh.find_hot_spot([0, 0.5, 0.1, 0.2], attraction=["SMA"]))
