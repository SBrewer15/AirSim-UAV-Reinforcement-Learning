

import airsim
import os
import numpy as np
import pandas as pd
import math
import time
import cv2
import nb_files.nb_Utilities as util

class Environment:
    def __init__(self, vehicle_name,  home= (0,0,0), maxz=120, maxspeed=8.33, episode_time=9): # change episode_time to 900 seconds (15 minutes)
        self.vehicle_name =vehicle_name
        self.home=home
        self.maxz=maxz
        self.maxspeed=maxspeed
        self.episode_time=episode_time
        self.reward=0
        self.sz=(224,224)

    def StartTime(self, start, episode):
        self.start=start; self.end=start
        self.episode=episode

    def GetTime(self, end): self.end=end

    def make_env(self):
         # connect to the simulator
        self.client=airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name) # enable API control on Drone0
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)        # arm Drone

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()  # let take-off
        self.client.moveToPositionAsync(self.home[0], self.home[1], self.home[2], 5, # 5m/s
                                   vehicle_name=self.vehicle_name).join()  #(note the inverted Z axis)

        self.SetSegmentID()
        print("Honey I'm home")
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        # initialize gps data to dataframe here
        self.df_gps=util.GPShistory(self.client.getMultirotorState().kinematics_estimated.position,
                           self.client.getMultirotorState().kinematics_estimated.linear_velocity,
                           0, time.time_ns(), self.vehicle_name, self.sz, self.maxspeed)
        time.sleep(2)
        responses = self.get_observations()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        # future add ignore collisions until drone is at home position
        self.client.moveToPositionAsync(self.home[0], self.home[1], self.home[2], 5,
                                   vehicle_name=self.vehicle_name).join()
        print("Honey I'm home")
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        # initialize gps data to dataframe here
        self.df_gps=util.GPShistory(self.client.getMultirotorState().kinematics_estimated.position,
                           self.client.getMultirotorState().kinematics_estimated.linear_velocity,
                           0, time.time_ns(), self.vehicle_name, self.sz, self.maxspeed)
        time.sleep(0.5)
        responses = self.get_observations()


    def get_position(self):
        return self.client.getMultirotorState().kinematics_estimated.position
    def get_velocity(self):
        return self.client.getMultirotorState().kinematics_estimated.linear_velocity

    def step(self, action):

        # convert actions to movement
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.quad_offset = self.interpret_action(action)
        x_vel , y_vel, z_vel=self.governor()

        self.client.moveByVelocityAsync(x_vel , y_vel, z_vel, 1 ).join() # movement interval needs to be understood
        # add check to keep below max speed
        #print( quad_state)
        time.sleep(0.5)
        # get images
        next_state=self.next_state()

        #reward=self.reward()

        return 'next_state', 'reward', self.done(), 'info'


    def done(self):
        done=False
        # episode ends if time runs out or collision
        self.deltaTime=self.end - self.start
        timeisUp = True if self.deltaTime>=self.episode_time else False
        if timeisUp: print('tic toc ... time is up')
        if self.client.simGetCollisionInfo().has_collided: print('Oh fiddlesticks, we just hit something...')
        if timeisUp or self.client.simGetCollisionInfo().has_collided: done = True

        return done

    def reward(self):
        reward=self.deltaTime
        if self.client.simGetCollisionInfo().has_collided: reward-=1000

        # reward for finding obstruction +100
            # convert found flag to True
        # reward for road and powerline follow needed
        # Drone update to pandas?
            # Get distance between drone if less than 100 meters update dataframe
            # penalize duplicate locations
            # bonus for update with obstructions
            # Method to prevent constant updating/ reward maximization of swarm
        # penalize entering no fly zone
        return reward

    def get_observations(self):
        self.responses= self.client.simGetImages([ airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, pixels_as_float=True),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False), # for reward
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar, pixels_as_float=True)]) # for reward

    def next_state(self):
        # convert to size 224x224 images
        # get depth image and front camera
        responses = self.get_observations()
        self.df_gps.appendGPShistory(self.client.getMultirotorState().kinematics_estimated.position,
                           self.client.getMultirotorState().kinematics_estimated.linear_velocity,
                           0, time.time_ns(), self.vehicle_name)
        #print(self.df_gps.df.head())
        deltaTime=self.end - self.start
        img_depth=util.byte2np_Depth(self.responses[0], Save=True, path='data', filename='Front_center_DepthPlanarS')
        img_rgb=util.byte2np_RGB(self.responses[1], Save=True, path='data', filename='Front_center_RGBS')
        img_seg=util.byte2np_RGB(self.responses[2], Save=True, path='data', filename=f'Front_center_SegS')
        #img_gps=self.df_gps.GPS2image(Save=False, path='data', filename='GPS')


        # previous 3 graysacle image frames
        # Depth of current frame
        # update map images
            # store positions, velocities, vehicle name, T/F Obstruction found
            # Map to have no fly zones

        # return stacked image frames to n-channel image
        return 'state_'

    def interpret_action(self, action):
        """Interprete action"""
        scale= 2
        assert action in np.arange(0,7), 'action must be between 0-6'
        if action == 0:
            self.quad_offset = (0, 0, 0)
        elif action == 1:
            self.quad_offset = (scale, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scale, 0)
        elif action == 3:
            self.quad_offset = (0, 0, scale)
        elif action == 4:
            self.quad_offset = (-scale, 0, 0)
        elif action == 5:
            self.quad_offset = (0, -scale, 0)
        elif action == 6:
            self.quad_offset = (0, 0, -scale)

        return self.quad_offset

    def governor(self):
        '''converts z velocity to zero if above limit,
        returns x, y, and z velocities in proportion to current but under max velocity'''

        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        z_pos = self.client.getMultirotorState().kinematics_estimated.position.z_val
        x_vel=quad_vel.x_val + self.quad_offset[0]
        y_vel=quad_vel.y_val + self.quad_offset[1]
        z_vel=quad_vel.z_val + self.quad_offset[2]

        if z_pos <-self.maxz: z=0
        rss=math.sqrt(x_vel*x_vel+y_vel*y_vel+z_vel*z_vel)
        if rss>self.maxspeed:
            x_vel=x_vel/rss*self.maxspeed
            y_vel=y_vel/rss*self.maxspeed
            z_vel=z_vel/rss*self.maxspeed

        return x_vel , y_vel, z_vel


    def addWeather(self, weather= False, fog=0.0, rain=0.0, dust=0.0,
                        snow=0.0, leaf=0.0, Roadwetness=0.0, wind=(0,0,0)):
        self.client.simEnableWeather(weather)
        if weather==True:
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog);
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain);
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow);
            self.client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, leaf);
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, Roadwetness);
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, dust);
            #self.client.simSetWind( airsim.Vector3r(wind[0],wind[1],wind[2])) this can't find the set wind function

    def SetSegmentID(self):
        df_mesh=pd.read_csv('data/meshes.csv', index_col=0)
        # turn off segmentation mesh for everything in neighborhood environment
        for mesh in df_mesh['0'].unique():
            success=self.client.simSetSegmentationObjectID(mesh, 0, True);
            #print(mesh, success)
        #airsim.wait_key('Press any key to reset')

        for i, mesh in enumerate(['grass', 'road', 'tree','stop', 'sphere', 'driveway','car', 'power']):
            assert i<255, 'too many meshs'
            #print(mesh,self.client.simGetSegmentationObjectID(f"{mesh}[\w]*"))
            success=self.client.simSetSegmentationObjectID(f"{mesh}[\w]*", i+1, True);
            print(mesh, ' is ', i+1, success)
