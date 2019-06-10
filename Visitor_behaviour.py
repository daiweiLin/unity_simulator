import numpy as np


class Visitor_behaviour:

    def __init__(self, num_visitor, node_number):
        self.num_visitors = num_visitor
        self.node_number = node_number

    def step(self, observation):
        '''
        :return: [visitor1.x, visitor1.y, visitor2.x, visitor2.y, ... visitorN.x, visitorN.y]
        '''
        # Find turned-on LEDs and randomly choose
        x = []
        y = []
        for i in range(self.node_number):
            if observation[i] > 0:
                x.append(observation[self.node_number + i * 2])
                y.append(observation[self.node_number + i * 2 + 1])

        if len(x) > 1:
            visitor_actions = []
            for i in range(self.num_visitors):
                random = np.random.randint(low=0, high=len(x) - 1)
                visitor_actions.append(x[random])
                visitor_actions.append(y[random])
                # print("visitor {} find light at {}".format(i, visitor_actions))
        elif len(x) == 1:
            visitor_actions = [x[0], y[0]] * self.num_visitors
            # print("visitor find light at {}".format(visitor_action))
        else:
            visitor_actions = np.random.uniform(low=-1, high=1, size=self.num_visitors * 2) * np.array(
                [12.5, 7.5] * self.num_visitors)

        return visitor_actions

