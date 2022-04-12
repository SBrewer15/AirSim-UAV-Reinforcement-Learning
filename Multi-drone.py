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
N_episodes=901 # 100 (20 second) episodes, 100 (35 second) episodes,
               # 100 (40 second) episodes, 100 (40 seconds) episodes nonlinear reward epsilon start 0.5
               # 100 (40 seconds) episodes nonlinear reward epsilon start 0.5 added back segmentation of road sky and not road
               # 100 (60 seconds) episodes nonlinear reward epsilon start 0.25 added back full segmentation
               # position 4
               # 300 (60 seconds) episodes nonlinear reward epsilon start 0.75  just roads segmenation

episode_time=60
ln=f'Z_&_Nonlinear_Road_Follow-Reward_w_BackTrack_Penalty_Pos4_{episode_time}s'
vehicle_name0="Drone0"
vehicle_name1="Drone1"
sz=(224,224)
env_name=f'Neighborhood'
algo=f'DDQNAgent_{ln}'

df_home=pd.DataFrame([[100.,0.], [75.,-15.], [0.,-100.], [0,100.], [-100.,0.]], columns=['x','y'])
df_nofly=pd.DataFrame([[75.,0., 20.],[75.,-40.,20.],[0.,-75.,20.],[0.,75.,20.],[-75.,0., 20.]], columns=['x','y','radius']) # Note no fly zone index cooresponds to home position
df_summary=pd.DataFrame([], columns=['Episode', 'Score', 'Average Score', 'Best Score',
                                     'steps', 'Model Saved', 'Epsilon', 'Dropout', 'Vehicle Name'])

env0=Environment(vehicle_name=vehicle_name0,
                home=(0, 0, -5), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env0.make_env()

env1=Environment(vehicle_name=vehicle_name1,
                home=(0, 0, -5), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env1.make_env()

agent0 = DDQN(gamma=0.99, epsilon=1, lr=0.00001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=2500, eps_min=0.1,
             batch_size=64, replace=500, eps_dec=5e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

agent1 = DDQN(gamma=0.99, epsilon=1, lr=0.00001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=2500, eps_min=0.1,
             batch_size=64, replace=500, eps_dec=5e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

load_name= 'Neighborhood_DDQNAgent_Z_&_Road_Follow-Reward_w_BackTrack_Penalty_Pos1_50s' # 'Neighborhood_600s_DDQNAgent_2022-03-26'
if load_name is not None:
    agent0.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval', suffex='')
    agent0.q_next.load_previous_checkpoint(f'models/{load_name}_q_next', suffex='')
    agent0.memory.load_memory_buffer(load_name)
    df_summary=pd.read_csv(f'data/{load_name}.csv')
    best_score = df_summary.loc[df_summary.index[-1], 'Best Score']
    n_steps = df_summary.loc[df_summary.index[-1],'steps']
    epsilon  = df_summary.loc[df_summary.index[-1],'Epsilon']
    episode_start  = df_summary.loc[df_summary.index[-1],'Episode']+1
    epsilon=0.75
    #best_score = -np.inf
    #n_steps = 0;  episode_start=0
    agent0.set_epsilon(epsilon)
    print(f'Loaded Old Model data, Epsilon {epsilon:0.2f} Drone Replay Buffer {agent0.memory.memory_counter}')
    agent1.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval', suffex='')
    agent1.q_next.load_previous_checkpoint(f'models/{load_name}_q_next', suffex='')

else:
    best_score = -np.inf
    n_steps = 0;  episode_start=0

Start=dt.datetime.now()
Episode_lst=[e for e in range(N_episodes)]
for episode in Episode_lst[episode_start:]:
    env0.ResetNoFlyZone()
    env1.ResetNoFlyZone()

    idx=4#df_home.sample().index[0]
    env0.Newhome(list(df_home.loc[idx])+[np.random.choice([-5.,-75.,-40.,-20.,-30.], p=[0.1,0.1, 0.3, 0.3,0.2])])
    env0.NewNoFlyZone([list(df_nofly.loc[idx]), [50,125,20]])#, [30,-110,20], [-30,-110,20], [0,-150,20]]) # currently only one no fly zone but method allows a list of them

    env1.Newhome(list(df_home.loc[idx]+1)+[np.random.choice([-5.,-75.,-40.,-20.,-30.], p=[0.1,0.1, 0.3, 0.3,0.2])])
    env1.NewNoFlyZone([list(df_nofly.loc[idx]), [50,125,20]])#, [30,-110,20], [-30,-110,20], [0,-150,20]]) # currently on

    score = 0;
    done=False
    state0=env0.reset()
    state1=env1.reset()

    drone_pos_dict, drone_gps_dict= None, None
    env0.StartTime(time.time(), episode)
    df_print=pd.DataFrame([False]*(int(episode_time/60)+1), columns=['Printed'])
    while done==False:

        action0 = agent0.choose_action(state)

        next_state0, reward, done, info = env0.step(action0, drone_pos_dict, drone_gps_dict)
        score += reward
        #print(action, score)
        agent0.store_transition(state0, action0,reward, next_state0, int(done))
        agent0.learn()
        #drone1
        action1 = agent1.choose_action(state)
        next_state1, _, _, _ = env1.step(action1, drone_pos_dict, drone_gps_dict)

        end=dt.datetime.now()
        print("* ",action0, f'{reward:0.2f}', info, f'{end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

        state0= next_state0
        state1= next_state1
        n_steps += 1

        # for multiple drones
        # make drone position dictionary
        drone_pos_dict = {vehicle_name0: env0.get_position(), vehicle_name1: env1.get_position()}
        # make drone gps dictionary
        drone_gps_dict = {vehicle_name0: env0.df_gps.getDataframe(), vehicle_name1: env1.df_gps.getDataframe()}


        env0.GetTime(time.time())
        #episode stats once a minute
        if 0.8>env0.deltaTime/60%1 <0.2 and df_print.loc[int(env0.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
            end=dt.datetime.now()
            print(f'Total Steps: {n_steps}, Time {env0.deltaTime/60:0.1f}(min) Score {score: 0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

            df_print.loc[int(env0.deltaTime/60), 'Printed']=True # prints only once
        if score <-100000: done = True # if the score is too bad kill the episode
    ## ************************* episode has ended ************************** ##
    avg_score = np.mean(df_summary.loc[df_summary.index[-9:],'Score'].to_list()+[score])
    df_summary.loc[len(df_summary)]= [episode, score, avg_score, best_score, n_steps,
                                      True if avg_score > best_score and episode>3 else False,
                                      agent0.epsilon, agent0.dropout,vehicle_name]
    # Save Model
    if avg_score > best_score and episode>3: # if episode legnth is not constant then this needs to change to score/second?
        agent0.save_models()
        best_score = avg_score
        # load other drone with best model
        agent1.q_eval.load_previous_checkpoint(f'models/{algo}_q_eval', suffex='')
        agent1.q_next.load_previous_checkpoint(f'models/{algo}_q_next', suffex='')

    # print summary
    print(df_summary.tail(5).T)
    # Save Stuff
    agent0.memory.save_memory_buffer()
    filename=f'{env_name}_{algo}'
    #env.df_gps.saveGPS2csv(f'data/GPS/gps_data_{vehicle_name}_episode{episode}_{filename}.csv')
    df_summary.to_csv(f'data/{filename}.csv', index=False)
    #saves copy of model on these intervals
    if episode in [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800,900, 1000,1100,1200,1300,1400,1500]:
        agent0.q_eval.save_weights_On_EpisodeNo(episode)
        agent0.q_next.save_weights_On_EpisodeNo(episode)


# Fin
print(f'Started at {Start.strftime("%a %b %d, %y")} at {Start.strftime("%H:%M")}')
end=dt.datetime.now()
print(f'Finished at {end.strftime("%A %B %d, %Y")} at {end.strftime("%H:%M")}')
