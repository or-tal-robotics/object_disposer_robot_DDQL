#!/usr/bin/env python

import gym
import numpy as np
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from imagetranformer import transform
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
import cv2
import tensorflow as tf
from datetime import datetime
import sys
from std_msgs.msg import Int16
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os.path
from os import path


MAX_EXPERIENCE = 50000 #50000
MIN_EXPERIENCE = 2000 #was 500 and before 5000 2000
TARGET_UPDATE_PERIOD = 50000
IM_SIZE = 64
LASER_SIZE = 720
LASER_MIN = 0.1
LASER_MAX = 10
K = 3 #4
n_history = 4


def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y

def shuffle_models(models, target_models, experience_replay_buffer,):
    models_temp = []
    target_models_temp = []
    experience_replay_buffer_temp = []
    Nm = len(models)
    for ii in range(Nm):
        idx = np.random.randint(0,Nm-ii)
        models_temp.append(models.pop(idx))
        target_models_temp.append(target_models.pop(idx))
        experience_replay_buffer_temp.append(experience_replay_buffer.pop(idx))
    return models_temp, target_models_temp, experience_replay_buffer_temp

def play_ones(
            env,
            total_t,
            experience_replay_buffer_object_disposer_robot,
            object_disposer_robot_model,
            target_models_object_disposer_robot,
            gamma,
            batch_sz,
            epsilon,
            epsilon_change,
            epsilon_min):
    
    t0 = datetime.now()
    obs = env.reset()
    img_obs=obs[0]
    lsr_obs=obs[1]


  
    

    #obs_small1 = image_transformer.transform(img_obs[0][0], sess)
    #obs_small2 = image_transformer.transform(img_obs[0][1], sess)
    #state_prey1 = np.stack([obs_small1] * n_history, axis = 2)
    #state_prey2 = np.stack([obs_small2] * n_history, axis = 2)
    
    #state_prey_laser = np.stack([laser_obs[0]] * n_history, axis = 1)

    obs_small = transform(img_obs,  [IM_SIZE, IM_SIZE])
    #lsr_small = transform(lsr_obs,  [IM_SIZE, IM_SIZE])


    
    #new addition 10.3.20
    
    state_object_disposer_robot = np.stack([obs_small] * n_history, axis = 2)
    state_object_disposer_robot_lsr= np.stack([lsr_obs] * n_history, axis = 2)



    
    loss = None
    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0
    record = True
    done = False
    loss = 0
    while not done:
        
        if total_t % TARGET_UPDATE_PERIOD == 0:
            # target_models_prey.copy_from(prey_model)
            # target_models_predator.copy_from(predator_model)
            target_models_object_disposer_robot.copy_from(object_disposer_robot_model)
            print("model is been copied!")
        #action.append(prey_model.sample_action(state_prey1, state_prey2, epsilon))
        action = object_disposer_robot_model.sample_action(state_object_disposer_robot,state_object_disposer_robot_lsr, epsilon)
        obs, reward, done, _ = env.step(action)
        img_obs = obs[0]
        lsr_obs = obs[1]
        
        episode_reward += reward

        

        obs_small = transform(img_obs,  [IM_SIZE, IM_SIZE])
        #lsr_small = transform(lsr_obs,  [IM_SIZE, IM_SIZE])
        
        
        
        next_state_object_disposer_robot,next_state_object_disposer_robot_lsr= update_state(state_object_disposer_robot, obs_small,state_object_disposer_robot_lsr,lsr_obs)
        experience_replay_buffer_object_disposer_robot.add_experience(action, obs_small,lsr_obs, reward, done)

        t0_2 = datetime.now()

        loss += learn(object_disposer_robot_model, target_models_object_disposer_robot, experience_replay_buffer_object_disposer_robot, gamma, batch_sz)
        #    if ii == 1:
        #        loss = learn(predator_model, target_models_predator, experience_replay_buffer_predator, gamma, batch_sz)
        dt = datetime.now() - t0_2
        
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        
        
        state_object_disposer_robot = next_state_object_disposer_robot #camera
        state_object_disposer_robot_lsr = next_state_object_disposer_robot_lsr #laser
        
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        
    return total_t, episode_reward, (datetime.now()-t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon , loss


if __name__ == '__main__':
    print("Starting training!!!")
    

    

    rospy.init_node('predator_prey_training_node',
                    anonymous=True, log_level=rospy.WARN)
    episode_counter_pub = rospy.Publisher('/episode_counter', Int16)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")   
    obs = env.reset()
    
    # Create the Gym environment
    print("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('dql_robot')
    rospy.loginfo("Monitor Wrapper started")
    last_time_steps = np.ndarray(0)
    gamma = rospy.get_param("/turtlebot2/gamma")
    K = rospy.get_param("/turtlebot2/n_actions")
    
    batch_sz = 32
    num_episodes = rospy.get_param("/turtlebot2/nepisodes")
    total_t = 0
    start_time = time.time()
    highest_reward = 0
    train_idxs = [0,1]
    skip_intervel = 50
    epsilon = rospy.get_param("/turtlebot2/epsilon")
    epsilon_min = rospy.get_param("/turtlebot2/epsilon_min")
    #epsilon_change = (epsilon - epsilon_min) / 100000 150000 300000
    epsilon_change = (epsilon - epsilon_min) / 100000
    
    # experience_replay_buffer_prey = ReplayMemory_multicamera(frame_height = IM_SIZE, fram_width=IM_SIZE, agent_history_lenth=n_history)
    # prey_model = DQN_prey(
    #     K = K,
    #     scope="prey_model",
    #     image_size1=IM_SIZE,
    #     image_size2=IM_SIZE,
    #     laser_size = LASER_SIZE,
    #     lase_min = LASER_MIN,
    #     laser_max = LASER_MAX,
    #     n_history = n_history
    #     )
    # target_models_prey = DQN_prey(
    #     K = K,
    #     scope="prey_target_model",
    #     image_size1=IM_SIZE,
    #     image_size2=IM_SIZE,
    #     laser_size = LASER_SIZE,
    #     lase_min = LASER_MIN,
    #     laser_max = LASER_MAX,
    #     n_history = n_history
    #     )

    # experience_replay_buffer_predator = ReplayMemory(frame_height = IM_SIZE, fram_width=IM_SIZE,agent_history_lenth=n_history)
    # predator_model = DQN_predator(
    #     K = K,
    #     scope="predator_model",
    #     image_size=IM_SIZE,
    #     n_history = n_history
    #     )
    # target_models_predator = DQN_predator(
    #     K = K,
    #     scope="predator_target_model",
    #     image_size=IM_SIZE,
    #     n_history = n_history
    #     )   

    experience_replay_buffer_object_disposer_robot = ReplayMemory(frame_height = IM_SIZE, fram_width=IM_SIZE,agent_history_lenth=n_history)
    object_disposer_robot_model = DQN(
        K = K,
        image_size=IM_SIZE
        )
    target_models_object_disposer_robot = DQN(
        K = K,
        image_size=IM_SIZE
        ) 



    episode_rewards = np.zeros(num_episodes)
    episode_lens = np.zeros(num_episodes)
    obs = env.reset()



    print("Initializing experience replay buffer...")
    obs = env.reset()
    
    for i in range(MIN_EXPERIENCE):
        
        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        img_obs=obs[0]
        lsr_obs=obs[1]

        obs_small = transform(img_obs, [IM_SIZE,IM_SIZE])
        #lsr_small = transform(lsr_obs, [IM_SIZE,IM_SIZE])



        experience_replay_buffer_object_disposer_robot.add_experience(action,obs_small,lsr_obs, reward, done)
        if done:
            obs = env.reset()

    
    print("Done! Starts Training newwww!!")
            
    #print "11111"
    #with open('/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/src/results/results.csv', 'w') as newFile:
    with open(rospack.get_path('dql_robot')+'/src/results/results.csv', 'w') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(['Episode', 'Reward','Epsilon','Avg Reward'])
        
            
    t0 = datetime.now()
    for i in range(num_episodes):
        msg_data = Int16()
        msg_data.data = i
        episode_counter_pub.publish(msg_data)

        if i % skip_intervel == 0:
            if train_idxs == [0]:
                train_idxs = [1]
            else:
                train_idxs = [0]

    

        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon, loss = play_ones(
                env,
                total_t,
                experience_replay_buffer_object_disposer_robot,
                object_disposer_robot_model,
                target_models_object_disposer_robot,
                gamma,
                batch_sz,
                epsilon,
                epsilon_change,
                epsilon_min)
        last_100_avg = []
        
        # episode_rewards[ii,i] = episode_reward[ii]
        # last_100_avg.append(episode_rewards[ii,max(0,i-100):i+1].mean())
        episode_rewards[i] = episode_reward
        last_100_avg.append(episode_rewards[max(0,i-100):i+1].mean())
        episode_lens[i] = num_steps_in_episode
        
        #i_to_csv=np.append(i,axis=0)
        #reward_to_csv=np.append(reward,axis=0)
        #avg_reward_to_csv=np.append(last_100_avg,axis=0)

        #dict = {'episode': i_to_csv, 'reward': reward_to_csv, 'avg reward': avg_reward_to_csv}
        #df = pd.DataFrame(dict)
        #df.to_csv(r'\result.csv', index=False) 

        
        

        print("Episode:", i ,
                "Duration:", duration,
                "Num steps:", num_steps_in_episode,
                "Reward:", episode_reward,
                "Training time per step:", "%.3f" %time_per_step,
                "Avg Reward : "+str(last_100_avg),
                "Epsilon:", "%.3f"%epsilon)
        sys.stdout.flush()

        #with open('/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/src/results/results.csv', 'a') as newFile:
        with open(rospack.get_path('dql_robot')+'/src/results/results.csv', 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow([i, episode_reward,epsilon,last_100_avg])

    print("Total duration:", datetime.now()-t0)

    
    
    
    #if  not os.path.isfile('/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/src/results/results.csv'):
        #print "11111"
        #with open('/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/src/results/results.csv', 'w') as newFile:
            #newFileWriter = csv.writer(newFile)
            #newFileWriter.writerow(['Episode', 'Reward','epsilon'])
            #newFileWriter.writerow([i, episode_reward,epsilon])
    #else:
    #print "22222"
    #with open('/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/src/results/results.csv', 'a') as newFile:
        #newFileWriter = csv.writer(newFile)
        #newFileWriter.writerow([i, episode_reward,epsilon])


    y1 = smooth(episode_rewards)
    #y2 = smooth(episode_rewards[1,:])

    plt.plot(y1, label='object_disposer_robot')
    #plt.plot(y2, label='predator')

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
    env.close()    



