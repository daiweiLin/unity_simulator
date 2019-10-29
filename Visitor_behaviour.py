import numpy as np


class Visitor_behaviour:

    def __init__(self, num_visitors, epsilon=0):
        self.num_visitors = num_visitors
        self.epsilon = epsilon
        # self.visitor_stay_time = stay_time # seconds
        self.node_number = None
        self.dist_matrix = None


        # self.visitor_start_ts = np.zeros(self.num_visitors, dtype=np.float64)
        self.visitor_prev_dest = [None]*self.num_visitors
        self.visitor_arrived_at_node = [False]*self.num_visitors

    def setup(self, coordinates):
        self.node_number = int(len(coordinates) / 2)
        self.dist_matrix = self._cal_node_distance(coordinates, normalize=True)

    @staticmethod
    def _cal_node_distance(coordinates, normalize=True):
        num_nodes = int(len(coordinates)/2)
        dis_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)

        for i in range(num_nodes):
            start_point = np.array([coordinates[i*2], coordinates[i*2+1]])
            for j in range(i, num_nodes):
                end_point = np.array([coordinates[j*2], coordinates[j*2+1]])
                dis_matrix[i, j] = np.linalg.norm(end_point-start_point)
                dis_matrix[j, i] = dis_matrix[i, j] # because of symmetry

        if normalize:
            # normalize distance so that the minimal distance is 1.
            # This is to ensure center nodes are more important than surrounding nodes
            dis_sort = np.sort(dis_matrix.flatten())
            min_dis = 0.0
            for d in dis_sort:
                if d != 0:
                    min_dis = d
                    break
            assert min_dis != 0.0, "Minimum distance between nodes is 0.0. Double check distance matrix."
            if min_dis < 1:
                dis_matrix = dis_matrix / min_dis

        # Make diagonal element equal to 1. This is to avoid numerical error in _find_hot_spot()
        for i in range(num_nodes):
            dis_matrix[i,i] = 1
        # print(dis_matrix)
        return dis_matrix

    @staticmethod
    def _cal_node_visitor_distance(v_coordinates, n_coordinates):
        distance = np.zeros((int(len(n_coordinates)/2),))
        for idx in range(len(distance)):
            d = np.linalg.norm(np.array(v_coordinates) - np.array(n_coordinates[idx*2:idx*2+2]))
            # Avoid numerical error
            if d == 0.0:
                d = 1e-6
            distance[idx] = d
        return distance

    def _find_hot_spot(self, observation, v_coordinates, n_coordinates, timeout=False, prev_dest=None):
        """
        Find the spot with most activities in the area
        The score is calculated using the sum of inverse of distances times observation values. Then it is divided by
        the distance between visitor and nodes.
        :return: index of selected spot if exists, return None o.w.
        """
        observation = np.array(observation)
        heat = np.zeros(self.node_number)
        for i in range(self.node_number):
            distance = self.dist_matrix[i, :]
            #distance[i] = 1 # change this to avoid numerical error (moved into _cal_node_distance())
            heat[i] = np.sum(observation * np.reciprocal(distance))
        '''
        Only consider those nodes with LED turned ON.
        Prevent visitors from going to a OFF node with many turned ON nodes around it.
        Then divide Heat by the distance between visitor and each node to select the nearest interesting spot.
        '''
        heat = heat * (observation > 0)
        #print("filtered heat=\n{}".format(heat))
        v_n_distance = self._cal_node_visitor_distance(v_coordinates, n_coordinates)
        heat = heat / v_n_distance

        #==============================#
        # Select spot w/ highest score #
        #==============================#
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
                            is_arrived_1, is_timeout_1, is_arrived_2, is_timeout_2,... is_arrived_m,is_timeout_m,
                            v_x1, v_y1, ...v_xm, v_ym
                            ingame_time]
                            ============================================================================================
                            LEDn                      : LED intensity
                            xn, yn                    : LED's (x,y) coordinates,
                            is_arrived_m, is_timeout_m: whether visitor arrives at destination and whether visitor time
                                                        out.
                            v_xm, v_ym                : visitor current coordinates
                            ============================================================================================
                            3*n + 4*m + 1 elements in total.
                            n = number of nodes;
                            m = number of visitors;

        :return:  [visitor1.x, visitor1.y, visitor2.x, visitor2.y, ... visitorN.x, visitorN.y]
        """
        visitor_actions = list()
        obs = observation[0 : self.node_number]
        # print("visitor observation :{}".format(obs))
        node_positions = observation[self.node_number:self.node_number*3]

        is_arrv = observation[self.node_number*3:self.node_number*3 + self.num_visitors*2][::2]
        is_timeout = observation[self.node_number*3:self.node_number*3 + self.num_visitors*2][1::2]
        # if np.sum(is_timeout) > 0:
        #     print("Timeout:{}".format(is_timeout))
        visitor_coords = observation[self.node_number*3 + self.num_visitors*2:self.node_number*3 + self.num_visitors*4]

        for v in range(self.num_visitors):
            dest = None
            if is_timeout[v] == 1:
                # Time out: the visitor has spent too much time trying to get to destination
                dest = self._find_hot_spot(observation=obs, v_coordinates=visitor_coords[v*2:v*2+2], n_coordinates=node_positions,
                                           timeout=True, prev_dest=self.visitor_prev_dest[v])
                self.visitor_prev_dest[v] = dest

            elif is_arrv[v]:
                prev_dest = self.visitor_prev_dest[v]
                if np.random.random_sample() < self.epsilon:
                    dest = None
                    self.visitor_prev_dest[v] = None
                else:
                    if prev_dest is None:
                        # Visitor just arrived at a random position, so he/she moves to next location
                        dest = self._find_hot_spot(observation=obs, v_coordinates=visitor_coords[v*2:v*2+2],
                                                   n_coordinates=node_positions)
                        self.visitor_prev_dest[v] = dest
                    else:
                        # Visitor just arrived at a node, so he/she wants to stay as long as the light is ON
                        # When light is turned OFF, the visitor select a new destination
                        if obs[prev_dest] <= 0:
                            dest = self._find_hot_spot(observation=obs, v_coordinates=visitor_coords[v*2:v*2+2],
                                                       n_coordinates=node_positions)
                            self.visitor_prev_dest[v] = dest
                        else:
                            dest = self.visitor_prev_dest[v]

            else:
                dest = self.visitor_prev_dest[v]

            # Convert dest(int) into coordinates
            if dest is not None:
                # Found a destination

                # print("Destination = Node{}".format(dest))
                visitor_actions.append(node_positions[2 * dest])
                visitor_actions.append(node_positions[2 * dest + 1])
            else:
                # Random select a position in space
                random_dest = np.random.uniform(low=-1, high=1, size=2) * np.array([10.0, 6.5])
                visitor_actions = visitor_actions + random_dest.tolist()
                # print("Destination = Random {}".format(random_dest))


        return visitor_actions


if __name__ == "__main__":
    # This section is for testing Visitor_behaviour Class

    visitor_bh = Visitor_behaviour(num_visitors=1)
    visitor_bh.setup(coordinates=[0, 0, 0, 1.5, 1.5, 0, 1.5, 1.5])
    print("\ndistance matrix:")
    print(visitor_bh.dist_matrix)

    print("\nDistance between visitor and lights:")
    print(visitor_bh._cal_node_visitor_distance(v_coordinates=[0, 0], n_coordinates=[0, 0, 0, 1.5, 1.5, 0, 1.5, 1.5]))

    print("\nHot spot:")
    print(visitor_bh._find_hot_spot(observation=[0, 0.5, 0.1, 0.2], v_coordinates=[0, 0]))

