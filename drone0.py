# drone0.py
import airsim
import numpy as np
from nb_files.nb_env import Environment
import nb_files.nb_Utilities as util
from nb_files.nb_Agent import DDQN
import time
import os

util.set_seed(42)

N_episodes=1
episode_time=10
vehicle_name="Drone0"
sz=(224,224)
env=Environment(vehicle_name=vehicle_name, home=(15, -3, -7), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env.make_env()

agent = DDQN(gamma=0.99, epsilon=1.0, lr=0.0001,
             input_dims=((4,)+sz),
             n_actions=7, mem_size=1000, eps_min=0.1,
             batch_size=32, replace=10000, eps_dec=1e-5,
             chkpt_dir='models/', algo='DDQNAgent',
             env_name='Neighborhood')

for episode in range(N_episodes):
    score = 0
    done=False
    state=env.reset()

    np.save(f'data/FirstArray_{episode}', state)
    env.StartTime(time.time(), episode)
    while done==False:
        #action =np.random.randint(0, high=3, size=1, dtype=int)[0]
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        score += reward

        if done:
            np.save(f'data/LastArray_{episode}', next_state)



        env.GetTime(time.time())
    print(f'Episode Number {episode+1} Complete')
    #env.df_gps.saveGPS2csv(f'data/gps_data_{vehicle_name}_episode{episode+1}.csv')
