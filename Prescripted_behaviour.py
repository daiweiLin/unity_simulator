# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:47:11 2019

@author: Daiwei Lin
"""
import time
import numpy as np
import math

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
    def __init__(self, sma_num, led_num, moth_num, para=None):
        self.sma_num = sma_num
        self.led_num = led_num
        self.moth_num = moth_num

        self.sma_act = False
        self.led_act = False
        self.moth_act = False

        self.sma_start_t = 0.0
        self.led_start_t = 0.0
        self.moth_start_t = 0.0

        self.sma_val = [0.0]*sma_num
        self.led_val = [0.0]*led_num
        self.moth_val = [0.0]*moth_num

        # ====================#
        # Tunable parameters #
        # ====================#
        if para is not None:
            self.sma_dur = para['sma_dur']
            self.led_dur = para['led_dur']
            self.moth_dur = para['moth_dur']
        else:
            self.sma_dur = 5
            self.led_dur = 2
            self.moth_dur = 2

    def node_step(self):
        """
        Perform one step of one node and returns actions
        :return: led*num_led, sma*num_sma
        """

        curr_t = time.time()

        sma_duration = curr_t - self.sma_start_t
        if sma_duration <= 5.0:
            for i in range(self.sma_num):
                self.sma_val[i] = sma_duration / 5.0
        else:
            for i in range(self.sma_num):
                self.sma_val[i] = 0
            self.sma_act = False
        # print("SMA : {}".format(self.sma_val))

        led_duration = curr_t - self.led_start_t
        if led_duration <= 2.0:
            for i in range(self.led_num):
                self.led_val[i] = led_duration / 2.0
        else:
            for i in range(self.led_num):
                self.led_val[i] = 0
            self.led_act = False

        # print("LED : {}".format(self.led_val))

        return self.led_val, self.sma_val #self.moth_val

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


class Sculpture:
    def __init__(self, node_num, sma_num, led_num, moth_num):

        self.node_list = list()
        self.num_nodes = node_num

        self.sma_num = sma_num
        self.led_num = led_num
        self.moth_num = moth_num

        for _ in range(node_num):
            self.node_list.append(Node(sma_num, led_num, moth_num))

        print("Sculpture initialized.")
        self.sculpture_info()

        # create a chain that defines the structure of sculpture
        # Used in propagation
        self.chain = list()
        for i in range(int(node_num/2)):
            self.chain.append([2*i, 2*i+1])

    def sculpture_step(self):
        '''
        Perform one step of the sculpture and returns actions of all actuators
        :return: [led*num_led*num_node, sma*num_sma*num_node]
        '''

        # loop through all nodes
        sma_action = list()
        led_action = list()
        for node in self.node_list:
            node_sma_action, node_led_action = node.node_step()
            sma_action = sma_action + node_sma_action
            led_action = led_action + node_led_action

        return sma_action + led_action

    def activate_local_reflex(self, index, act_list):
        '''

        :param index: Node index
        :param act_list: list of actuators to activate in the node
        '''
        for act in act_list:
            self.node_list[index].activate(act)

    def sculpture_info(self):
        print("{} nodes".format(self.num_nodes))
        print("Each node has {} sma, {} led, {} moth".format(self.sma_num, self.led_num, self.moth_num))


class Behaviour:
    def __init__(self, sculpture):
        self.state = 'idle'  # either idle or active
        self.sculpture = sculpture
        self.state_timer = time.time()
        self.idle_timer = time.time()
        self.propagate_timer = time.time()

        self.propagation_list = list()

        print("Behaviour initialized.")

    def step(self, observation):
        # update system by one step

        # <state transition>
        for obs in observation:
            if obs > 0:
                self.state = 'active'
                self.state_timer = time.time()
                break

        if self.state == 'active' and time.time() - self.state_timer > 10:
            print("Enter Idle state.")
            self.state = 'idle'
        # <end of state transition>

        print("\n{} Step ----------".format(self.state))
        if self.state == 'idle':
            # randomly select one node
            if time.time() - self.idle_timer > 1:
                self.idle_timer = time.time()

                rand_idx = np.random.randint(0, self.sculpture.num_nodes)
                rand_actuator = np.random.choice(['led', 'sma'], 1)[0]
                print("(IDLE)Activate {} on node{}".format(rand_actuator, rand_idx))
                self.sculpture.activate_local_reflex(rand_idx, [rand_actuator])
        else:
            # actuate at the triggered node and propagate
            trigger_nodes = observation > 0
            for idx in range(len(observation)):
                if trigger_nodes[idx]:
                    print("(ACTIVE)Activate {} on node{}".format(['led', 'sma', 'moth'], idx))
                    self.sculpture.activate_local_reflex(idx, ['led', 'sma', 'moth'])
                    self.create_propagation_list(idx)

            # activate the nodes for propagation
            if len(self.propagation_list) > 0:
                if time.time()-self.propagate_timer > 1:
                    self.propagate_timer = time.time()
                    print("Propagation list {}".format(self.propagation_list))
                    for node in self.propagation_list[0]:
                        print("(ACTIVE)(P)Activate {} on node{}".format(['led', 'sma', 'moth'], node))
                        self.sculpture.activate_local_reflex(node, ['led', 'sma', 'moth'])
                    self.propagation_list.pop(0)


        action = self.sculpture.sculpture_step()
        # print("action = {}".format(action))
        return action

    def create_propagation_list(self, node_index):
        # find location
        start_point = -1
        for i in range(len(self.sculpture.chain)):
            if node_index in self.sculpture.chain[i]:
                start_point = i
                break
        assert start_point >= 0, "No starting point found for propagation."

        p_list = list()
        i = 1
        while True:
            if node_index-i >= 0 and node_index+i <= self.sculpture.num_nodes-1:
                p_list.append([node_index-i, node_index+i])
            elif node_index-i >= 0:
                p_list.append([node_index-i])
            elif node_index+i <= self.sculpture.num_nodes-1:
                p_list.append([node_index+i])
            else:
                break
            i += 1
        self.propagation_list = p_list
        # print("Propagation list {}".format(self.propagation_list))


if __name__ == '__main__':
    ROM_sculpture = Sculpture(node_num=4, sma_num=2, led_num=1, moth_num=1)
    behaviour = Behaviour(ROM_sculpture)

    for i in range(1000):
        observation = np.array([0,0,0,0])
        if i % 20 == 19:
            observation = np.array([0,0,1,0])

        action =behaviour.step(observation)
        print(action)
        time.sleep(0.2)
