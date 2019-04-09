# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:47:11 2019

@author: Daiwei Lin
"""
#include <Python.h>
import time        

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

class Scupture:
    def __init__(self, actuators):
        self.sma_list = list()
        self.led_list = list()
        self.moth_list = list()
        # actuators is a dictionary, i.e. {'sma':12, 'led':12}
        for act_type, num_actuators in actuators.items():
            if act_type == 'sma':
                for _ in range(num_actuators):
                    self.sma_list.append(Actuator(act_type))
            elif act_type == 'led':
                for _ in range(num_actuators):
                    self.led_list.append(Actuator(act_type))
            elif act_type == 'moth':
                for _ in range(num_actuators):
                    self.moth_list.append(Actuator(act_type))
            else:
                raise ValueError("Invalid actuator type: {}".format(act_type))
        

class behaviour:
    def __init__(self, sculpture):
        self.state = 'idle'  # either idle or active
        self.sculpture = sculpture
        self.state_timer = time.time()
    
    def step(self):
        # update system by one step
        if self.state == 'idle':
            # do something
        else:
            # do something

    def actuate(self):
    
    def propagate(self):
        for i in self.sculpture.sma_list:
            self.actuate()
