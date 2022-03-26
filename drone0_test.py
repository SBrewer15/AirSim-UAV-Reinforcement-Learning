# drone0_test.py
# Drone flies with no randomn actions
import airsim
import numpy as np
import pandas as pd
from nb_files.nb_env import Environment
import nb_files.nb_Utilities as util
from nb_files.nb_Agent import DDQN
import time
import os
import datetime as dt

util.set_seed(42)

pd.options.display.float_format = "{:,.3f}".format
tm=dt.datetime.now().strftime("%Y-%m-%d")

episode_time=10
vehicle_name="Drone0"
sz=(224,224)
env_name=f'Neighborhood_test'
algo=f'DDQNAgent'

df_home=pd.DataFrame([[15,0], [73,0], [110,0], [0,0],[0,120],[0,-120]], columns=['x','y'])
df_home['z']=np.random.randint(-60, high=-12, size=len(df_home), dtype=int)
df_home=df_home+np.random.random_sample((len(df_home), 3)) *10-5

env=Environment(vehicle_name=vehicle_name,
                home=(0, 0, -10), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env.make_env()



agent = DDQN(gamma=0.99, epsilon=0.0, lr=0.0001,
             input_dims=((7,)+sz),
             n_actions=7, mem_size=5000, eps_min=0.0,
             batch_size=256, replace=500, eps_dec=1e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

load_name='Neighborhood_600s_DDQNAgent_2022-03-25'
agent.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval', suffex='_epsiode_50')
agent.q_next.load_previous_checkpoint(f'models/{load_name}_q_next', suffex='_epsiode_50')
#Neighborhood_600s_DDQNAgent_2022-03-23_q_eval_epsiode_100


#env.client.moveToPositionAsync(15, -3, -30, 5, vehicle_name=env.vehicle_name).join()
#env.client.hoverAsync(vehicle_name=env.vehicle_name).join()
#env.get_observations()
#img_seg=util.byte2np_Seg(env.responses[3], Save=True, path='data', filename=f'Bottom_center_Seg_z30')

score = 0; done=False; n_steps = 0; episode=0

state=env.reset()
env.StartTime(time.time(), episode)
df_print=pd.DataFrame([False]*(int(episode_time/60)+1), columns=['Printed'])
act_lst=[]
info_lst=['started']

idx=df_home.sample().index[0]
env.Newhome(list(df_home.loc[idx]))
env.NewNoFlyZone([list(df_home.loc[df_home[df_home.index!=0].sample().index[0], ['x','y']])+[20]])
while done==False:

    action = agent.choose_action(state)
    if n_steps==0: act_lst.append(action)

    next_state, reward, done, info = env.step(action)#, drone_pos_dict, drone_gps_dict)
    score += reward
    act_lst.append(action)
    info_lst.append(info)

    state= next_state
    n_steps += 1

    print(action, reward, info)

    env.GetTime(time.time())
    if 0.8>env.deltaTime/60%1 <0.2 and df_print.loc[int(env.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
        end=dt.datetime.now()
        print(f'Total Steps: {n_steps}, Time {env.deltaTime/60:0.1f}(min) Score {score:0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

        df_print.loc[int(env.deltaTime/60), 'Printed']=True # prints only once
    if score <-100000: done = True # if the score is too bad kill the episode
episode='test'
if episode in ['test']:
    agent.q_eval.save_weights_On_EpisodeNo(episode)
    agent.q_next.save_weights_On_EpisodeNo(episode)

frame=env.df_gps.df.copy()
frame['action']=act_lst
frame['info']= info_lst
frame.to_csv(f'data/{load_name}_test.csv', index=False)

env.ResetNoFlyZone()
# Fin
end=dt.datetime.now()
print(f'Finished at {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')
