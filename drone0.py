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
episode_time=900
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

scores, eps_history, steps_array = [], [], []
n_steps = 0
load_checkpoint = False
best_score = -np.inf

for episode in range(N_episodes):
    score = 0
    done=False
    state=env.reset()

    np.save(f'data/FirstArray_{episode}', state)
    env.StartTime(time.time(), episode)
    while done==False:

        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)
        score += reward
        print(action, score)
        if not load_checkpoint:
            agent.store_transition(state, action,reward, next_state, int(done))
            agent.learn()
        state= next_state
        n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score



        if done:
            np.save(f'data/LastArray_{episode}', next_state)
        env.GetTime(time.time())
    print(f'Episode Number {episode+1} Complete. Final Score {score}, Number of steps taken: {n_steps}')
    env.df_gps.saveGPS2csv(f'data/gps_data_{vehicle_name}_episode{episode+1}_DDQN.csv')
    eps_history.append(agent.epsilon)
    print(n_steps)
