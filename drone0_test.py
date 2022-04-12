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

episode_time=90
vehicle_name="Drone0"
sz=(224,224)
env_name=f'Neighborhood_test'
algo=f'DDQNAgent'

df_home=pd.DataFrame([[100.,0.], [75.,-15.], [0.,-100.], [0,100.], [-100.,0.]], columns=['x','y'])
df_nofly=pd.DataFrame([[75.,0., 20.],[75.,-40.,20.],[0.,-75.,20.],[0.,75.,20.],[-75.,0., 20.]], columns=['x','y','radius']) # Note no fly zone index cooresponds to home position

env=Environment(vehicle_name=vehicle_name,
                home=(0, 0, -30), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env.make_env()

agent = DDQN(gamma=0.99, epsilon=0.0, lr=0.0001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=5000, eps_min=0.0,
             batch_size=256, replace=500, eps_dec=1e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

load_name='Neighborhood_DDQNAgent_Z_&_Road_Follow-Reward_w_BackTrack_Penalty_Pos1_76s'
agent.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval', suffex='')#_epsiode_10
agent.q_next.load_previous_checkpoint(f'models/{load_name}_q_next', suffex='')
#Neighborhood_600s_DDQNAgent_2022-03-23_q_eval_epsiode_100


#env.client.moveToPositionAsync(15, -3, -30, 5, vehicle_name=env.vehicle_name).join()
#env.client.hoverAsync(vehicle_name=env.vehicle_name).join()
#env.get_observations()
#img_seg=util.byte2np_Seg(env.responses[3], Save=True, path='data', filename=f'Bottom_center_Seg_z30')
df_hist=pd.DataFrame([],columns=['x_position','y_position','z_position','x_velocity','y_velocity','z_velocity',
                                 'Reward','time_stamp','vehicle_name','action','episode','info'])

episode=10
for episode in range(1,11):
    score = 0; done=False; n_steps = 0;
    env.ResetNoFlyZone()
    idx=1
    env.Newhome(list(df_home.loc[idx])+[-30])#list(df_home.loc[idx])+[-30])
    env.NewNoFlyZone([list(df_nofly.loc[idx])]) # currently only one no fly zone but method allows a list of them

    state=env.reset()
    env.StartTime(time.time(), episode)
    df_print=pd.DataFrame([False]*(int(episode_time/60)+1), columns=['Printed'])
    act_lst=[]
    info_lst=['started']

    Start=dt.datetime.now()

    while done==False:

        action = agent.choose_action(state)
        if n_steps==0: act_lst.append(action)

        next_state, reward, done, info = env.step(action)#, drone_pos_dict, drone_gps_dict)
        score += reward
        act_lst.append(action)
        info_lst.append(info)

        state= next_state
        n_steps += 1

        end=dt.datetime.now()
        print(f"*Action: {action}, Episode: {episode}, Reward: {reward:0.2f}", info, f'{env.deltaTime:0.2f}s Time: {end.strftime("%H:%M")}')

        env.GetTime(time.time())
        if 0.8>env.deltaTime/60%1 <0.2 and df_print.loc[int(env.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
            end=dt.datetime.now()
            print(f'Total Steps: {n_steps}, Time {env.deltaTime/60:0.1f}(min) Score {score:0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

            df_print.loc[int(env.deltaTime/60), 'Printed']=True # prints only once
        if score <-100000: done = True # if the score is too bad kill the episode


    frame=env.df_gps.df.copy()
    frame['action']=act_lst
    frame['episode']= episode
    frame['info']= info_lst

    df_hist=frame.append(df_hist)
    df_hist.to_csv(f'data/{load_name}_test.csv', index=False)


    print(f'Total Steps: {n_steps}, Time {env.deltaTime/60:0.1f}(min) Score {score:0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')
# Fin
print(f'Started at {Start.strftime("%a %b %d, %y")} at {Start.strftime("%H:%M")}')
end=dt.datetime.now()
print(f'Finished at {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')
