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

#util.set_seed(42)
pd.options.display.float_format = "{:,.3f}".format
tm=dt.datetime.now().strftime("%Y-%m-%d")

N_episodes=500
episode_time=900
vehicle_name="Drone0"
sz=(224,224)
env_name=f'Neighborhood_{episode_time}s2'
algo=f'DDQNAgent_{tm}'

df_nofly=pd.DataFrame([], columns=['x','y','radius']) #37.2, 50.9, 15], [175,60,20]

env=Environment(vehicle_name=vehicle_name, df_nofly=df_nofly,
                home=(15, -3, -30), maxz=120,
                maxspeed=8.33,episode_time=episode_time, sz=sz)
env.make_env()

agent = DDQN(gamma=0.99, epsilon=0.840, lr=0.0001,
             input_dims=((5,)+sz),
             n_actions=7, mem_size=5000, eps_min=0.1,
             batch_size=256, replace=500, eps_dec=1e-4,
             chkpt_dir='models/', algo=algo,
             env_name=env_name)

# training history
# Neighborhood_900s2_DDQNAgent_2022-03-17 start at epsilon 1.0 to 0.950
# Neighborhood_900s_DDQNAgent_2022-03-17 start at  epsilon 0.950 to 0.908
# Neighborhood_900s3_DDQNAgent_2022-03-17 start at epsilon 0.908 to 0.840
# Neighborhood_900s_DDQNAgent_2022-03-18 start at epsilon 0.840

# for curiculum learning
agent.q_eval.load_previous_checkpoint('models/Neighborhood_900s_DDQNAgent_2022-03-18_q_next')
agent.q_next.load_previous_checkpoint('models/Neighborhood_900s_DDQNAgent_2022-03-18_q_next')
#print('Loaded Old Model')

best_score = -np.inf; n_steps = 0
df_summary=pd.DataFrame([], columns=['Episode', 'Score', 'Average Score', 'Best Score',
                                     'steps', 'Model Saved', 'Epsilon', 'Dropout', 'Vehicle Name'])

for episode in range(N_episodes):
    score = 0;
    done=False
    state=env.reset()
    #env.client.moveToPositionAsync(15, -3, -110, 5, vehicle_name=env.vehicle_name).join()
    #env.client.hoverAsync(vehicle_name=env.vehicle_name).join()
    #env.get_observations()
    #print('Total Reward', env.Calculate_reward())

    #np.save(f'data/FirstArray_{episode}', state)
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

        state= next_state
        n_steps += 1

        # for multiple drones
        # make drone position dictionary
        #pos=env.get_position()
        #drone_pos_dict = {vehicle_name: env.get_position()}
        # make drone gps dictionary
        #drone_gps_dict = {vehicle_name: env.df_gps.getDataframe()}

        if score <-100000: done = True # if the score is too bad kill the episode
        #if done: #np.save(f'data/LastArray_{episode}', next_state)
        env.GetTime(time.time())
        if 0.8>env.deltaTime/60%1 <0.2 and df_print.loc[int(env.deltaTime/60), 'Printed']==False: # prints resutls every minute-ish
            print(f'Total Steps: {n_steps}, Time {env.deltaTime:0.1f}s Score {score: 0.1f}')
            df_print.loc[int(env.deltaTime/60), 'Printed']=True # prints only once
    ## episode has ended
    avg_score = np.mean(df_summary.loc[df_summary.index[-49:],'Score'].to_list()+[score])

    df_summary.loc[len(df_summary)]= [episode, score, avg_score, best_score, n_steps,
                                      True if avg_score > best_score else False,
                                      agent.epsilon, agent.dropout,vehicle_name]

    # Save Model
    if avg_score > best_score: # if episode legth is not constant then this needs to change to score/second
        agent.save_models()
        best_score = avg_score


    print(df_summary.tail(5).T)

    # Save Stuff
    env.df_gps.saveGPS2csv(f'data/gps_data_{vehicle_name}_episode{episode}_{algo}.csv')
    filename=f'{env_name}_{algo}'
    df_summary.to_csv(f'data/{filename}.csv', index=False)
    util.plot_Reward(df_summary, 'plots', filename)
