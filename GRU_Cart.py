''' This code is based off of an old tutorial from pythonprogramming.net, which was my first exposure to reinforcment learning.
This code however is more of an experiment, so a lot of it is changed and customized, as it was easier to do that than just rewrite the whole thing 
from scratch. This architecture usually gets a perfect 200 in the cartpole environment.
'''


import tflearn 
from tflearn.layers.recurrent import gru
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter


LR=1e-3
env=gym.make('CartPole-v0')
env.reset()
goal_steps=500
score_requirement=100
initial_games=20000
dropout_rate=0.8
num_epochs=5

def initial_population():
	training_data=[] #Our position and observations
	scores=[] #The scores corresponding to our positions.
	accepted_scores=[] #The scores that meet a certain threshhold.
	for _ in range(initial_games): #Now we iterate through however many games we want.
		score=0
		game_memory=[] #Information about the environment
		prev_observation=[] #the last observation we made.
		for _ in range(goal_steps): #
			action=random.randrange(0,2) #Choose a random action.
			observation, reward, done, info=env.step(action) #This makes us take the action.
			if len(prev_observation)>0: #This appends our previous action to a list, then stores the new one.
				game_memory.append([prev_observation, action])
			prev_observation=observation
			score+=reward
			if done: break
		if score >= score_requirement: #Here is the reinforcement step. If we do good (reach the threshhold) we wanna remember what we did.
			accepted_scores.append(score)
			for data in game_memory:
				if data[1]==1: #Converts to a one hot array.
					output=[0,1]
				elif data[1]==0:
					output=[1,0]
				training_data.append([data[0], output])
		env.reset() #Here we save and reset our score.
		scores.append(score)
	return training_data
	
	
def neural_network_model(input_size):
	network=input_data(shape=[None, input_size, 1], name='input')
	network=gru(network, 128, return_seq=True)
	network=gru(network, 256)
	network=batch_normalization(network)
	network=fully_connected(network, 256, activation='relu')
	network=dropout(network, dropout_rate)
	network=fully_connected(network, 256, activation='relu')
	network=dropout(network, dropout_rate)
	network=fully_connected(network, 2, activation='softmax')
	network=regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	model=tflearn.DNN(network)
	return model

	
	
def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1) #We reformat our data
	y = [i[1] for i in training_data]
	if not model:
		model = neural_network_model(input_size = len(X[0]))
	model.fit(X, y, n_epoch=num_epochs, snapshot_step=500, show_metric=True, run_id='openai_learning') #This is how our model is trained and by what parameters.
	return model



training_data = initial_population()			
model = train_model(training_data)
scores = []
choices = []
for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs)==0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
		choices.append(action)
		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		score+=reward
		if done: break
	scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
