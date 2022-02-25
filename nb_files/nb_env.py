

import airsim

import numpy as np
import pandas as pd

class Environment:
    def __init__(self, vehicle_name,  home= (0,0,0), maxz=120, maxspeed=8.33):
        self.vehicle_name =vehicle_name
        self.home=home
        self.maxz=maxz
        self.maxspeed=maxspeed


    def make_env(self):
         # connect to the simulator
        self.client=airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name) # enable API control on Drone0
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)        # arm Drone

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()  # let take-off
        self.client.moveToPositionAsync(self.home[0], self.home[1], self.home[2], 5, # 5m/s
                                   vehicle_name=self.vehicle_name).join()  #(note the inverted Z axis)
        #print('Home')
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        # future add ignore collisions until drone is at home position
        self.client.moveToPositionAsync(self.home[0], self.home[1], self.home[2], 5,
                                   vehicle_name=self.vehicle_name).join()
        #print('Home')
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()



    def get_position(self):
        return self.client.getMultirotorState().kinematics_estimated.position
    def get_velocity(self):
        return self.client.getMultirotorState().kinematics_estimated.linear_velocity

    def step(self, action):
        # get images
        # convert actions to movement
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.quad_offset = self.interpret_action(action)
        MOVEMENT_INTERVAL = 1
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            1 ).join()
        collision = self.client.simGetCollisionInfo().has_collided
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        #print( quad_state)
        return 'next_state', 'reward', 'done', 'info'

    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 2
        assert action in np.arange(0,7)
        if action == 0:
            self.quad_offset = (0, 0, 0)
        elif action == 1:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            self.quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action == 5:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            self.quad_offset = (0, 0, -scaling_factor)

        return self.quad_offset
