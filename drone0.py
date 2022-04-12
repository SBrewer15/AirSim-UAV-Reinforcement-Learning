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
N_episodes=1751 # 100 (20 second) episodes, 100 (35 second) episodes,
               # 100 (40 second) episodes, 100 (40 seconds) episodes nonlinear reward epsilon start 0.5
               # 100 (40 seconds) episodes nonlinear reward epsilon start 0.5 added back segmentation of road sky and not road
               # 100 (60 seconds) episodes nonlinear reward epsilon start 0.25 added back full segmentation



episode_time=76
ln=f'Z_&_Road_Follow-Reward_w_BackTrack_Penalty_Pos1_{episode_time}s'
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
epsilon=0.25
agent = DDQN(gamma=0.99, epsilon=epsilon, lr=0.00001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=2500, eps_min=0.1,
             batch_size=32, replace=500, eps_dec=5e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

load_name= 'Neighborhood_DDQNAgent_Z_&_Road_Follow-Reward_w_BackTrack_Penalty_Pos1_76s'
if load_name is not None:
    agent.q_eval.load_previous_checkpoint(f'models/{load_name}_q_eval', suffex='')
    agent.q_next.load_previous_checkpoint(f'models/{load_name}_q_next', suffex='')
    agent.memory.load_memory_buffer(load_name)
    df_summary=pd.read_csv(f'data/{load_name}.csv')
    best_score = df_summary.loc[df_summary.index[-1], 'Best Score']
    n_steps = df_summary.loc[df_summary.index[-1],'steps']
    epsilon  = df_summary.loc[df_summary.index[-1],'Epsilon']
    episode_start  = df_summary.loc[df_summary.index[-1],'Episode']+1
    #epsilon=0.25
    #best_score = -np.inf
    #n_steps = 0;  episode_start=0
    agent.set_epsilon(epsilon)
    print(f'Loaded Old Model data, Epsilon {epsilon:0.2f} Drone Replay Buffer {agent.memory.memory_counter}')
else:
    best_score = -np.inf
    n_steps = 0;  episode_start=0

saved_epsilon=epsilon
explore_epsilon=1.; explore=False
Start=dt.datetime.now()
Episode_lst=[e for e in range(N_episodes)]
for episode in Episode_lst[episode_start:]:
    env.ResetNoFlyZone()

    idx=1
    z=np.random.choice([-50.,-75.,-40.,-20.,-30.], p=[0.1,0.1, 0.3, 0.3,0.2])
    env.Newhome(list(df_home.loc[idx])+[z])
    env.NewNoFlyZone([list(df_nofly.loc[idx])])#, [30,-110,20], [-30,-110,20], [0,-150,20]]) # currently only one no fly zone but method allows a list of them

    score = 0;
    done=False
    state=env.reset()

    drone_pos_dict, drone_gps_dict= None, None
    env.StartTime(time.time(), episode)
    df_print=pd.DataFrame([False]*(int(episode_time/60)+1), columns=['Printed'])
    while done==False:

        if (agent.epsilon<= 0.1)&(env.deltaTime>=episode_time-20):
            print(f'Exploring at end of episode. Explore-Epsilon: {explore_epsilon:0.3f}')
            explore=True
            agent.set_epsilon(explore_epsilon)
            action = agent.choose_action(state)
        else:
            explore=False
            action = agent.choose_action(state)


        next_state, reward, done, info = env.step(action)#, drone_pos_dict, drone_gps_dict)
        score += reward
        #print(action, score)
        agent.store_transition(state, action,reward, next_state, int(done))
        agent.learn()
        if explore:
            explore_epsilon=agent.epsilon
            agent.set_epsilon(saved_epsilon)
        else:
            saved_epsilon=agent.epsilon

        end=dt.datetime.now()
        print("* ",action, f'{reward:0.2f}', info, f'{env.deltaTime:0.2f}s Time: {end.strftime("%H:%M")}')

        state= next_state
        n_steps += 1

        env.GetTime(time.time())
        #episode stats once a minute
        if 0.8>env.deltaTime/60%1 <0.2 and df_print.loc[int(env.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
            end=dt.datetime.now()
            print(f'Total Steps: {n_steps}, Time {env.deltaTime/60:0.1f}(min) Score {score: 0.1f} {end.strftime("%a %b %d, %y")} at {end.strftime("%H:%M")}')

            df_print.loc[int(env.deltaTime/60), 'Printed']=True # prints only once
        if score <-100000: done = True # if the score is too bad kill the episode
    ## ************************* episode has ended ************************** ##
    avg_score = np.mean(df_summary.loc[df_summary.index[-9:],'Score'].to_list()+[score])
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
    if episode in [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800,900, 1000,1100,1200,1300,1400, 1500, 1600, 1750]:
        agent.q_eval.save_weights_On_EpisodeNo(episode)
        agent.q_next.save_weights_On_EpisodeNo(episode)


# Fin
print(f'Started at {Start.strftime("%a %b %d, %y")} at {Start.strftime("%H:%M")}')
end=dt.datetime.now()
print(f'Finished at {end.strftime("%A %B %d, %Y")} at {end.strftime("%H:%M")}')
