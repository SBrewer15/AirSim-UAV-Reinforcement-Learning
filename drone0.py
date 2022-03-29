# drone0.py
import airsim
import numpy as np
import pandas as pd
from nb_files.nb_env import Environment
import nb_files.nb_Utilities as util
from nb_files.nb_Agent import DDQN
import time
import os
import datetime as dt
#./AirSimNH.sh  -ResX=640 -ResY=480 -windowed
#util.set_seed(42)
pd.options.display.float_format = "{:,.3f}".format
tm=dt.datetime.now().strftime("%Y-%m-%d")
ln='_Z-Reward_&_Road_Follow'
N_episodes=301
episode_time=30
vehicle_name="Drone0"
sz=(224,224)
env_name=f'Neighborhood'
algo=f'DDQNAgent_{ln}'

df_home=pd.DataFrame([[100.,0.], [75.,-15.], [0.,-100.], [0,100.], [-100.,0.]], columns=['x','y'])
df_nofly=pd.DataFrame([[75.,0., 20.],[75.,-40.,20.],[0.,-75.,20.],[0.,75.,20.],[-75.,0., 20.]], columns=['x','y','radius']) # Note no fly zone index cooresponds to home position
df_summary=pd.DataFrame([], columns=['Episode', 'Score', 'Average Score', 'Best Score',
                                     'steps', 'Model Saved', 'Epsilon', 'Dropout', 'Vehicle Name'])

env=Environment(vehicle_name=vehicle_name,
                home=(0, 0, -5), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env.make_env()

agent = DDQN(gamma=0.99, epsilon=1, lr=0.0001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=2500, eps_min=0.1,
             batch_size=64, replace=500, eps_dec=1e-3,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

load_name= 'Neighborhood_DDQNAgent_Just_Z-Reward_Low&High' # 'Neighborhood_600s_DDQNAgent_2022-03-26'
if load_name is not None:
    agent.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval')
    agent.q_next.load_previous_checkpoint(f'models/{load_name}_q_next')
    #agent.memory.load_memory_buffer(load_name)
    #df_summary=pd.read_csv(f'data/{load_name}.csv')
    #best_score = df_summary.loc[df_summary.index[-1], 'Best Score']
    #n_steps = df_summary.loc[df_summary.index[-1],'steps']
    #epsilon  = df_summary.loc[df_summary.index[-1],'Epsilon']
    #episode_start  = df_summary.loc[df_summary.index[-1],'Episode']+1
    epsilon=1
    best_score = -np.inf
    n_steps = 0;  episode_start=0
    #agent.set_epsilon(0.75)#epsilon)
    print(f'Loaded Old Model data, Epsilon {epsilon:0.2f} Drone Replay Buffer {agent.memory.memory_counter}')
else:
    best_score = -np.inf
    n_steps = 0;  episode_start=0

#min_z=3; Z=30; Z_dec=0.99
#max_z=110; Z=30; Z_inc=1.01
Episode_lst=[e for e in range(N_episodes)]
for episode in Episode_lst[episode_start:]:
    env.ResetNoFlyZone()
    #if Z>min_z: Z=Z*Z_dec
    #else: Z=min_z
    #high=np.random.choice([0,1], p=[0.1, 0.9])
    #if high==1:
    #    if Z<max_z: Z=Z*Z_inc
    #    else: Z=max_z
    #    Zee=Z
    #else: Zee=7
    #env.Newhome([0,0,-1*Zee])

    idx=df_home.sample().index[0]
    env.Newhome(list(df_home.loc[idx])+[np.random.choice([-5.,-75.,-40.,-20.,-30.], p=[0.1,0.1, 0.3, 0.3,0.2])])
    env.NewNoFlyZone([list(df_nofly.loc[idx])]) # currently only one no fly zone but method allows a list of them

    score = 0;
    done=False
    state=env.reset()

    drone_pos_dict, drone_gps_dict= None, None
    env.StartTime(time.time(), episode)
    df_print=pd.DataFrame([False]*(int(episode_time/60)+1), columns=['Printed'])
    while done==False:

        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)#, drone_pos_dict, drone_gps_dict)
        score += reward
        #print(action, score)
        agent.store_transition(state, action,reward, next_state, int(done))
        agent.learn()
        print(action, f'{reward:0.2f}', info)

        state= next_state
        n_steps += 1

        # for multiple drones
        # make drone position dictionary
        #pos=env.get_position()
        #drone_pos_dict = {vehicle_name: env.get_position()}
        # make drone gps dictionary
        #drone_gps_dict = {vehicle_name: env.df_gps.getDataframe()}


        env.GetTime(time.time())
        #episode stats once a minute
        if 0.8>env.deltaTime/60%1 <0.2 and df_print.loc[int(env.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
            end=dt.datetime.now()
            print(f'Total Steps: {n_steps}, Time {env.deltaTime/60:0.1f}(min) Score {score: 0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

            df_print.loc[int(env.deltaTime/60), 'Printed']=True # prints only once
        if score <-100000: done = True # if the score is too bad kill the episode
    ## ************************* episode has ended ************************** ##
    avg_score = np.mean(df_summary.loc[df_summary.index[-19:],'Score'].to_list()+[score])
    df_summary.loc[len(df_summary)]= [episode, score, avg_score, best_score, n_steps,
                                      True if avg_score > best_score and episode>3 else False,
                                      agent.epsilon, agent.dropout,vehicle_name]
    # Save Model
    if avg_score > best_score and episode>3: # if episode legnth is not constant then this needs to change to score/second?
        agent.save_models()
        best_score = avg_score

    # print summary
    print(df_summary.tail(5).T)
    # Save Stuff
    agent.memory.save_memory_buffer()
    filename=f'{env_name}_{algo}'
    #env.df_gps.saveGPS2csv(f'data/GPS/gps_data_{vehicle_name}_episode{episode}_{filename}.csv')
    df_summary.to_csv(f'data/{filename}.csv', index=False)
    #saves copy of model on these intervals
    if episode in [10, 50, 100, 150, 200, 250, 300, 400, 500]:
        agent.q_eval.save_weights_On_EpisodeNo(episode)
        agent.q_next.save_weights_On_EpisodeNo(episode)


# Fin
end=dt.datetime.now()
print(f'Finished at {end.strftime("%A %B %d, %Y")} at {end.strftime("%H:%M")}')
