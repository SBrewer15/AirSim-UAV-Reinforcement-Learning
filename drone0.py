# drone0.py
import airsim
import numpy as np
from nb_files.nb_env import Environment
import nb_files.nb_Utilities as util

util.set_seed(42)

N_episodes=2
episode_time= 900 #seconds = 15 minutes

env=Environment(vehicle_name="Drone0", home=(15, -3, -7), maxz=120, maxspeed=8.33 )
env.make_env()
for epsiode in range(N_episodes):
    for action in np.random.randint(0, high=7, size=10, dtype=int):
        next_state, reward, done, info = env.step(action)
        #print(env.get_position().x_val)

    airsim.wait_key('Press any key to reset')
    env.reset()
    #print('Done', epsiode)
