
import airsim
import os
import numpy as np
import pandas as pd
import math
import time
import cv2
import nb_files.nb_Utilities as util

class Environment:
    def __init__(self, vehicle_name,  df_nofly, home= (0,0,0), maxz=120,
                    maxspeed=8.33, episode_time=900, sz=(224,224)): # change episode_time to 900 seconds (15 minutes)
        self.vehicle_name =vehicle_name
        self.home=home
        self.maxz=maxz
        self.maxspeed=maxspeed
        self.episode_time=episode_time
        self.reward=0
        self.sz=sz
        self.df_nofly=df_nofly
        self.vehicleNo=int(self.vehicle_name[-1])
        self.scale=5 #(meters per step)

    def StartTime(self, start, episode):
        self.start=start; self.end=start
        self.episode=episode

    def GetTime(self, end): self.end=end
    def Newhome(self, home): self.home=home

    def ChngEpisodeLnght(self, new_episode_time):
        self.episode_time=new_episode_time

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
        state = np.zeros((4,h,w))
        state[0]=util.DistanceSensor2Image(self.home[0],self.home[1],distance_dict=self.distance_dict,
                                            scale=self.scale, sz=self.sz, df_nofly=self.df_nofly)
        state[1]=util.byte2np_Seg(self.responses[2])
        state[2]=util.byte2np_Seg(self.responses[3])
        state[3]= util.initialGPS(self.home[0],self.home[1], sz=self.sz, df_nofly=self.df_nofly)
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
        #time.sleep(2)
        self.initializeState()
        self.deltaTime=0
        return (self.state-0.5)/0.5 # imagenet normalization for inception style network

    def get_position(self): return self.client.getMultirotorState().kinematics_estimated.position
    def get_velocity(self): return self.client.getMultirotorState().kinematics_estimated.linear_velocity

    def step(self, action, drone_pos_dict=None, drone_gps_dict=None):
        self.drone_pos_dict=drone_pos_dict
        self.drone_gps_dict=drone_gps_dict
        self.reward=0
        self.info=''
        # convert actions to movement
        self.quad_offset = self.interpret_action(action)
        #x_vel , y_vel, z_vel=self.governor()
        #self.client.moveByVelocityAsync(x_vel , y_vel, z_vel, 1).join() # movement interval needs to be understood
        x_pos = self.client.getMultirotorState().kinematics_estimated.position.x_val
        y_pos = self.client.getMultirotorState().kinematics_estimated.position.y_val
        z_pos = self.client.getMultirotorState().kinematics_estimated.position.z_val

        # get height below to limit crash landings
        if self.distance_dict['Z']<8:
            self.quad_offset=(self.quad_offset[0], self.quad_offset[1], 0)
            self.reward+=-10
            self.info+=' Too Low,'

        self.client.moveToPositionAsync(x_pos+self.quad_offset[0], y_pos+self.quad_offset[1],
                                        min(min(z_pos+self.quad_offset[2], self.maxz), -1), # z is negative
                                        min(5, self.maxspeed),
                                        vehicle_name=self.vehicle_name).join()

        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        #time.sleep(1)
        # get new images
        next_state=self.next_state()
        self.DetectObstruction()

        self.df_gps.appendGPShistory(self.client.getMultirotorState().kinematics_estimated.position,
                           self.client.getMultirotorState().kinematics_estimated.linear_velocity,
                           self.reward, time.time_ns(), self.vehicle_name)

        # Distance between drone if less than 100 meters update dataframe
        if drone_gps_dict is not None:
            for other_drone in drone_gps_dict.keys():
                if other_drone == self.vehicle_name: continue
                x_pos=(self.drone_pos_dict[self.vehicle_name].x_val-self.drone_pos_dict[other_drone].x_val)
                y_pos=(self.drone_pos_dict[self.vehicle_name].y_val-self.drone_pos_dict[other_drone].y_val)
                z_pos=(self.drone_pos_dict[self.vehicle_name].z_val-self.drone_pos_dict[other_drone].z_val)
                rss=math.sqrt(x_pos*x_pos+y_pos*y_pos+z_pos*z_pos)
                if rss<=100:
                    # this isn't the most effiecent way to do this
                    self.df_gps.df=self.df_gps.df.append(self.drone_gps_dict[other_drone])
                    self.df_gps.df.drop_duplicates(inplace=True)
                    self.df_gps.df.reset_index(inplace=True, drop=True)
        self.Calculate_reward()
        #print(self.reward)
        done=self.done()

        return next_state, self.reward, done, self.info

    def done(self):
        done=False
        # episode ends if time runs out, collision, or if reward is too high
        timeisUp = True if self.deltaTime>=self.episode_time else False
        if timeisUp:
            print('tic toc ... time is up')
            done = True
        if self.client.simGetCollisionInfo().has_collided:
            print('Oh fiddlesticks, we just hit something...')
            self.reward+= -5000
            self.info+=f' Collision,'
            done = True
        if self.distance_dict['Front']<1 or self.distance_dict['Back']<1 or self.distance_dict['Right']<1 or \
            self.distance_dict['Left']<1 or self.distance_dict['Z']<1:

            print('We are too close to crashing...')
            self.reward+= -4000
            self.info+=f' Collision Iminent,'
            done = True

        # add if the drone is too far from home

        if self.reward <-10000.: done = True# if the reward is too bad kill the episode
        return done

    def get_observations(self):
        self.responses= self.client.simGetImages([ airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, pixels_as_float=True),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                                          airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False), # for reward
                                          airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar, pixels_as_float=True)]) # for reward

        self.distance_dict=  {'Front': self.client.getDistanceSensorData(vehicle_name=self.vehicle_name, distance_sensor_name=f"DistanceFront_Drn{self.vehicleNo}").distance,
                              'Back': self.client.getDistanceSensorData(vehicle_name=self.vehicle_name, distance_sensor_name=f"DistanceBack_Drn{self.vehicleNo}").distance,
                              'Left': self.client.getDistanceSensorData(vehicle_name=self.vehicle_name, distance_sensor_name=f"DistanceLeft_Drn{self.vehicleNo}").distance,
                              'Right':self.client.getDistanceSensorData(vehicle_name=self.vehicle_name, distance_sensor_name=f"DistanceRight_Drn{self.vehicleNo}").distance,
                              'Z': util.Distance2Grnd(util.byte2np_Depth(self.responses[4], Normalize=False), self.sz, rng=10)}

    def next_state(self):
        self.deltaTime=self.end - self.start
        responses = self.get_observations()
        new_state = self.state.copy()

        x_position=self.client.getMultirotorState().kinematics_estimated.position.x_val
        y_position=self.client.getMultirotorState().kinematics_estimated.position.y_val
        z_position=self.client.getMultirotorState().kinematics_estimated.position.z_val

        new_state[0]=util.DistanceSensor2Image(x_position,y_position,distance_dict=self.distance_dict,
                                            scale=self.scale, sz=self.sz, df_nofly=self.df_nofly)
        new_state[1]=util.byte2np_Seg(self.responses[2])
        new_state[2]=util.byte2np_Seg(self.responses[3])
        new_state[3]=self.df_gps.GPS2image(x_position, y_position, self.df_nofly)
        self.state=new_state
        return (new_state-0.5)/0.5 # imagenet normalization for inception style network

    def Calculate_reward(self):
         # reward penalty is also in interpret action, step function to prevent collisions, and done
        x_position=self.client.getMultirotorState().kinematics_estimated.position.x_val
        y_position=self.client.getMultirotorState().kinematics_estimated.position.y_val
        z_position=self.client.getMultirotorState().kinematics_estimated.position.z_val

        # get reward for percent of road below
        roadReward=util.RoadBelowReward(util.byte2np_Seg(self.responses[3]), rng=50, reward=100)
        self.reward+=roadReward
        #print(self.reward)
        self.info+=f' Road Reward: {roadReward:0.1f},'

        # height
        hght=util.HghtReward(self.distance_dict['Z'])
        self.reward+=max(hght, -1000)
        self.info+=f' Height Penalty: {hght:0.1f},'

        backtrack=max(util.Penalty4Backtrack(self.df_gps.getDataframe(), drone_dict=self.drone_gps_dict,
                                      dist=20, penalty=-10, x=x_position, y=y_position), -500)
        self.reward+=backtrack
        self.info+=f' Backtrack Penalty: {backtrack:0.1f},'
        #print(f'Penalty for backtracking {backtrack}')
        if self.obstructionDetected: self.reward+=100

        # penalize drone distance from one another
        if self.drone_pos_dict is not None:
            for other_drone in self.drone_pos_dict.keys():
                if other_drone == self.vehicle_name: continue
                x_pos=(self.drone_pos_dict[self.vehicle_name].x_val-self.drone_pos_dict[other_drone].x_val)
                y_pos=(self.drone_pos_dict[self.vehicle_name].y_val-self.drone_pos_dict[other_drone].y_val)
                z_pos=(self.drone_pos_dict[self.vehicle_name].z_val-self.drone_pos_dict[other_drone].z_val)

                rss=math.sqrt(x_pos*x_pos+y_pos*y_pos+z_pos*z_pos)
                self.reward+=util.DroneDistanceReward(rss)
                self.info+=f' Drone Distance: {rss:0.1f},'
        # penalize entering no fly zone
        if len(self.df_nofly)>0:
            for idx in self.df_nofly.index:
                x_pos=x_position-self.df_nofly.loc[idx,'x']
                y_pos=y_position-self.df_nofly.loc[idx,'y']
                rss=math.sqrt(x_pos*x_pos+y_pos*y_pos)
                self.reward+=util.NoFlyZoneReward(rss-self.df_nofly.loc[idx,'radius'])
                self.info+=f" No Fly Zone Distance: {rss-self.df_nofly.loc[idx,'radius']:0.1f},"

        # penalize distance from home if greater than 5km
        x_pos=x_position-self.home[0]
        y_pos=y_position-self.home[1]
        z_pos=z_position-self.home[2]
        rss=math.sqrt(x_pos*x_pos+y_pos*y_pos+z_pos*z_pos)
        self.info+=f' Distance From Home: {rss:0.1f} (m),'
        if rss>=5000:
            self.reward+=-4000



    def interpret_action(self, action):
        """Interprete action"""
        scale= self.scale
        assert action in np.arange(0,7), 'action must be between 0-6'
        if action == 0: # you should alway be exploring
            self.quad_offset = (0, 0, 0)
            self.reward+=-10
        elif action == 1: # forward
            if self.distance_dict['Front'] <scale*1.1:
                self.quad_offset = (0, 0, 0)
                self.reward+=-10
                self.info+=' Stopped Front Impact,'
            else: self.quad_offset = (scale, 0, 0)
        elif action == 2: # right
            if self.distance_dict['Right']<scale*1.1:
                self.quad_offset = (0, 0, 0)
                self.reward+=-10
                self.info+=' Stopped Right Impact,'
            else: self.quad_offset = (0, scale, 0)
        elif action == 3: self.quad_offset = (0, 0, scale) # up
        elif action == 4: # back
            if self.distance_dict['Back']<scale*1.1:
                self.quad_offset = (0, 0, 0)
                self.reward+=-10
                self.info+=' Stopped Rear Impact,'
            else: self.quad_offset = (-scale, 0, 0)
        elif action == 5: # left
            if self.distance_dict['Left']<scale*1.1:
                self.quad_offset = (0, 0, 0)
                self.reward+=-10
                self.info+=' Stopped Left Impact,'
            else: self.quad_offset = (0, -scale, 0)
        elif action == 6: self.quad_offset = (0, 0, -scale)
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

    def DetectObstruction(self): self.obstructionDetected= False


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
