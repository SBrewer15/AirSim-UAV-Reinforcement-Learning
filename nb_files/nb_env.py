

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
        print("Initilized")
        self.SetSegmentID()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def initializeState(self):
        h,w=self.sz
        responses = self.get_observations()
        depth=util.byte2np_Depth(self.responses[0])
        seg=util.byte2np_Seg(self.responses[2])
        # GPS needs to be added
        state = np.random.random_sample((4,h,w))
        state[0]=seg
        state[1]=depth
        state[2]=depth
        state[3]=depth

        self.state=state

    def reset(self, weather= False, fog=0.0, rain=0.0, dust=0.0,
                        snow=0.0, leaf=0.0, Roadwetness=0.0, wind=(0,0,0)):
        self.addWeather(weather, fog, rain, dust, snow, leaf, Roadwetness, wind)
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
        time.sleep(2)
        self.initializeState()
        self.deltaTime=0
        return self.state

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
        
        # get images
        next_state=self.next_state()

        reward=self.Calculate_reward()

        return next_state, 'reward', self.done(), 'info'


    def done(self):
        done=False
        # episode ends if time runs out or collision
        timeisUp = True if self.deltaTime>=self.episode_time else False
        if timeisUp: print('tic toc ... time is up')
        if self.client.simGetCollisionInfo().has_collided: print('Oh fiddlesticks, we just hit something...')
        if timeisUp or self.client.simGetCollisionInfo().has_collided: done = True

        return done


    def get_observations(self):
        self.responses= self.client.simGetImages([ airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, pixels_as_float=True),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False), # for reward
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar, pixels_as_float=True)]) # for reward

    def next_state(self):
        self.deltaTime=self.end - self.start
        # convert to size 224x224 images
        # get depth image and front camera
        responses = self.get_observations()
        self.df_gps.appendGPShistory(self.client.getMultirotorState().kinematics_estimated.position,
                           self.client.getMultirotorState().kinematics_estimated.linear_velocity,
                           0, time.time_ns(), self.vehicle_name)

        depth=util.byte2np_Depth(self.responses[0])
        seg=util.byte2np_Seg(self.responses[2])
        # GPS needs to be added
        new_state = self.state.copy()
        new_state[0]=seg
        new_state[1]=depth
        new_state[2]=self.state[1]
        new_state[3]=self.state[2]
        self.state=new_state

        return new_state

    def Calculate_reward(self):

        img_seg=util.byte2np_Seg(self.responses[3], Save=False, path='data',
                                filename=f'Bottom_center_Seg_{self.episode}_{int(self.deltaTime*1000)}')
        img_depth=util.byte2np_Depth(self.responses[4], Save=False, path='data',
                                    filename='Bottom_center_DepthPlanarS', Normalize=False)
        #print('Is the road below:', util.isRoadBelow(img_seg, self.sz, rng=10))
        roadBelow=util.isRoadBelow(img_seg, self.sz, rng=10)
        z_ht=util.Distance2Grnd(img_depth, self.sz, rng=10)
        #print('Calculated Z height:',z_ht,'Actual Z Height:', self.client.getMultirotorState().kinematics_estimated.position.z_val)

        reward=0
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
            success=self.client.simSetSegmentationObjectID(mesh, 255, True);
            #print(mesh)
            #airsim.wait_key('Press any key to reset')

        Class2ID={'grass': 1, 'road':2, 'stop':4,
                  'sphere': 5, 'driveway':6, 'car': 7, 'power': 8,
                  'driveway':9, 'roof':30, 'wall': 11,
                  'street':13, 'path': 14, 'pool': 15, 'fence': 11,
                  'tree':3, 'birch': 3, 'oak': 3,'fir':3,
                  'hedge':17,'garden':27,
                  'cone': 16, 'porch': 18, 'house':19,  'chimney':20,  'garage':19,
                  'outer':19, # house walls
                  'lamp':20, 'monument':21, 'stairs':22, 'rock':23,
                  'bench':24, 'veranda':18,  'quadrotor':26}
        for mesh, ID in  Class2ID.items():
            #print(mesh,self.client.simGetSegmentationObjectID(f"{mesh}[\w]*"))
            success=self.client.simSetSegmentationObjectID(f"{mesh}[\w]*", ID, True);
            # print(mesh, ' is ', ID, success)
