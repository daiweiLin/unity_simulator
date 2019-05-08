# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:47:11 2019

@author: Daiwei Lin
"""
#include <Python.h>
import time
import numpy as np
'''
class Actuator:
    def __init__(self, act_type):
        assert act_type in ['sma','led','moth'], "Invalid acuator type {}, must be {}".format(act_type, ['sma','led','moth'])
        self.act_type = act_type

class Sensor:
    def __init__(self, sens_type):
        assert sens_type in ['ir'], "Invalid sensor type {}, must be {}".format(sens_type, ['ir'])
        self.sens_type = sens_type
    
    def read(self):
        # read sensor
        return None

'''
class Node:
    def __init__(self, sma_num, led_num, moth_num):
        self.sma_num = sma_num
        self.led_num = led_num
        self.moth_num = moth_num

        self.sma_act = False
        self.led_act = False
        self.moth_act = False

        self.sma_start_t = 0.0
        self.led_start_t = 0.0
        self.moth_start_t = 0.0

        self.sma_val = np.zeros((1,sma_num),dtype=np.float64)
        self.led_val = np.zeros((1,led_num),dtype=np.float64)
        self.moth_val = np.zeros((1,moth_num),dtype=np.float64)


    def node_step(self):
        curr_t = time.time()

        sma_duration = curr_t - self.sma_start_t
        if sma_duration <= 5.0:
            for i in range(self.sma_num):
                self.sma_val[i] = sma_duration/5.0
        else:
            for i in range(self.sma_num):
                self.sma_val[i] = 0
            self.sma_act = False
        print("SMA : ".format(self.sma_val.tolist()))

        led_duration = curr_t - self.led_start_t
        if sma_duration <= 2.0:
            for i in range(self.led_num):
                self.led_val[i] = led_duration / 2.0
        print("LED : ".format(self.led_val.tolist()))

    def activate(self, act_type):
        # start timer
        if act_type == 'sma':
            if not self.sma_act:
                self.sma_act = True
                self.sma_start_t = time.time()
        elif act_type == 'led':
            if not self.led_act:
                self.led_act = True
                self.led_start_t = time.time()
        elif act_type == 'moth':
            if not self.moth_act:
                self.moth_act = True
                self.moth_start_t = time.time()

        else:
            raise ValueError("Invalid actuator type: {}".format(act_type))


class Scupture:
    def __init__(self, node_num, sma_num, led_num, moth_num):
        # self.sma_list = list()
        # self.led_list = list()
        # self.moth_list = list()

        self.node_list = list()
        self.num_nodes = node_num

        self.sma_num = sma_num
        self.led_num = led_num
        self.moth_num = moth_num

        for _ in range(node_num):
            self.node_list.append(Node(sma_num,led_num,moth_num))

        print("Sculpture initialized.")
        self.scupture_info()
        # # actuators is a dictionary, i.e. {'sma':12, 'led':12}
        # for act_type, num_actuators in actuators.items():
        #     if act_type == 'sma':
        #         for _ in range(num_actuators):
        #             self.sma_list.append(Node(act_type))
        #     elif act_type == 'led':
        #         for _ in range(num_actuators):
        #             self.led_list.append(Node(act_type))
        #     elif act_type == 'moth':
        #         for _ in range(num_actuators):
        #             self.moth_list.append(Node(act_type))
        #     else:
        #         raise ValueError("Invalid actuator type: {}".format(act_type))

    def scupture_step(self):
        # loop through all nodes
        for node in self.node_list:
            node.node_step()

    def activate_local_reflex(self, index, act_type):
        self.node_list[index].activate(act_type)

    def scupture_info(self):
        print("{} nodes".format(self.num_nodes))
        print("Each node has {} sma, {} led, {} moth".format(self.sma_num, self.led_num, self.moth_num))


class Behaviour:
    def __init__(self, sculpture):
        self.state = 'idle'  # either idle or active
        self.sculpture = sculpture
        self.state_timer = time.time()
        self.idle_timer = time.time()

        print("Behaviour initialized.")
    
    def step(self):
        # update system by one step
        if self.state == 'idle':
            # randomly select one node
            if time.time() - self.idle_timer > 1:
                self.idle_timer = time.time()

                rand_idx = np.random.randint(0, self.sculpture.num_nodes)
                rand_actuator = np.random.choice(['led', 'sma'], 1)[0]
                print("Activate local reflex on node{}".format(rand_idx, rand_actuator))
                self.sculpture.activate_local_reflex(rand_idx, rand_actuator)

        else:
            # do something
            return


    def propagate(self):
        # for i in self.sculpture.sma_list:
        #     self.actuate()
        return 0




if __name__ == '__main__':
    ROM_sculpture = Scupture(1,2,1,1)
    behaviour = Behaviour(ROM_sculpture)

    while True:
        behaviour.step()