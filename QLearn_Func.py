'''

This is a quick and dirty Q-Learning function for exploring the concept. It runs the Frozen Lake environment, and can run most other environments
if the observation and action spaces are tweaked. That being said, there are much better algorithms for reinforcement learning, and a lot can be deployed
easily using Tensorforce, a RL library for Tensorflow.

'''

import numpy as np
import gym
import random

def QLearn(LR, min_LR, LR_decay, disc_factor, num_episodes, env_name, Static_LR=False):
	
	#We need to initialize our q matrix, and make an array to store our rewards in.
	reward_array=[]
	env=gym.make(env_name)
	q_table=np.zeros([env.observation_space.n, env.action_space.n])

	for i in range(num_episodes):
		state=env.reset() #Reset the environment to start a new episode.
		done=False
		total_reward=0
		
		
		if Static_LR==False: #This lets one experiment with decaying or static learning rates.
			eta=max(min_LR, LR*(LR_decay**(i//100)))
		if Static_LR==True:
			eta=LR
			
		#The following is where the "learning" really takes place.
		for k in range(100) :
			action=np.argmax(q_table[state,:]+np.random.randn(1, env.action_space.n)*(1./(i+1))) #We decide which action to take using a policy that explores at the start but buckles down as time foes on.
			new_state, reward, decision, _= env.step(action) #We extract the information from our action
			total_reward+=reward
			q_table[state, action]=(1-eta)*q_table[state, action]+eta*(reward+disc_factor*np.max(q_table[new_state,:])) #We then calculate the values of our q-table
			state=new_state #We then enter a new state.
			if done==True:
				break
		reward_array.append(total_reward)
		
		
	print("Score over time: " +  str(sum(reward_array)/num_episodes))


























