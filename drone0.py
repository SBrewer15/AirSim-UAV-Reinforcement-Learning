# drone0.py
import airsim
import numpy as np
from nb_files.nb_env import Environment
import nb_files.nb_Utilities as util
import time
import os

util.set_seed(42)

N_episodes=1
episode_time=100
vehicle_name="Drone0"

env=Environment(vehicle_name=vehicle_name, home=(15, -3, -7), maxz=120,
                maxspeed=8.33,episode_time=episode_time)
env.make_env()
for episode in range(N_episodes):

    done=False
    state=env.reset()

    filename = os.path.join('data', 'FirstArray')
    np.save(filename, state)
    env.StartTime(time.time(), episode)
    while done==False:
        action =np.random.randint(0, high=3, size=1, dtype=int)[0]
        next_state, reward, done, info = env.step(action)


        if done:
            filename = os.path.join('data', 'LastArray')
            np.save(filename, next_state)



        env.GetTime(time.time())
    print(f'Episode Number {episode+1} Complete')
    #env.df_gps.saveGPS2csv(f'data/gps_data_{vehicle_name}_episode{episode+1}.csv')
