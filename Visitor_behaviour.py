import numpy as np


class Visitor_behaviour:

    def __init__(self, num_visitors):
        self.num_visitors = num_visitors
        # self.visitor_stay_time = stay_time # seconds
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

    def find_hot_spot(self, observation, timeout=False, prev_dest=None):
        """
        Find the spot with most activities in the area
        The score is calculated using the sum of inverse of distances times observation values.
        :return: index of selected spot if exists, return None o.w.
        """
        observation = np.array(observation)
        heat = np.zeros(self.node_number)
        for i in range(self.node_number):
            distance = self.dist_matrix[i, :]
            distance[i] = 1 # change this to avoid numerical error
            heat[i] = np.sum(observation * np.reciprocal(distance))

        # print("heat={}".format(heat))
        # At least one hot spot
        if np.sum(heat > 0) > 0:
            if not timeout:
                return heat.argmax()
            else:
                #assert prev_dest is not None, "Must provide previous destination if visitor times out."
                sorted_dest = np.argsort(heat)
                for i in range(self.node_number - 1, -1, -1):
                    if prev_dest != sorted_dest[i]:
                        return sorted_dest[i]
        # No hot spot
        else:
            return None

    def step(self, observation):
        """
        NOTICE: is_timeout_x ONLY exist in multi-visitor case
        :param observation: [LED1, LED2, ... LEDn,
                            x1, y1, x2, y2, ... xn, yn,
                            is_arrived_1, (is_timeout_1), is_arrived_2, (is_timeout_2),... is_arrived_m,(is_timeout_m),
                            ingame_time]

                            3*n + 2*m + 1 elements in total.
                            n = number of nodes;
                            m = number of visitors;

        :return:  [visitor1.x, visitor1.y, visitor2.x, visitor2.y, ... visitorN.x, visitorN.y]
        """
        visitor_actions = list()
        obs = observation[0 : self.node_number]
        positions = observation[self.node_number:self.node_number*3]
        if self.num_visitors > 1:
            is_arrv = observation[self.node_number*3:self.node_number*3 + self.num_visitors*2][::2]
            is_timeout = observation[self.node_number*3:self.node_number*3 + self.num_visitors*2][1::2]
            # if np.sum(is_timeout) > 0:
            #     print("Timeout:{}".format(is_timeout))
        else:
            is_arrv = observation[self.node_number*3:self.node_number*3 + self.num_visitors]
            is_timeout = [False]

        for v in range(self.num_visitors):
            dest = None
            if is_timeout[v] == 1:
                # The visitor has spent too much time trying to get to destination
                dest = self.find_hot_spot(observation=obs, timeout=True, prev_dest=self.visitor_prev_dest[v])
                self.visitor_prev_dest[v] = dest

            elif is_arrv[v]:
                prev_dest = self.visitor_prev_dest[v]

                if prev_dest is None:
                    # Visitor just arrived at a random position, so he/she moves to next location
                    dest = self.find_hot_spot(observation=obs)
                    self.visitor_prev_dest[v] = dest
                else:
                    # Visitor just arrived at a node, so he/she wants to stay as long as the light is ON
                    if obs[prev_dest] <= 0:
                        dest = self.find_hot_spot(observation=obs)
                        self.visitor_prev_dest[v] = dest
                    else:
                        dest = self.visitor_prev_dest[v]

            else:
                # dest = self.find_hot_spot(observation=obs, visitor_at_node=None)
                dest = self.visitor_prev_dest[v]

            # Convert dest(int) into coordinates
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


        return visitor_actions


if __name__ == "__main__":
    # This section is for testing Visitor_behaviour Class

    visitor_bh = Visitor_behaviour(num_visitors=1, coordinates=[0,0, 0,1.5, 1.5,0, 1.5,1.5])
    print(visitor_bh.dist_matrix)
    print(visitor_bh.find_hot_spot([0, 0.5, 0.1, 0.2]))