''' This is a network that attempts to classify people's Myers-Briggs results based on their vocabulary. This problem is interesting as one might
expect the data to be insufficient for the task, as the MB test is notorious for being pseudoscientific. As one might expect, the network cannot 
learn a perfect, nor even particularly "good" mapping. However, the validation accuracy constantly achieves values from 30-50% which implies it can learn
some type of patchy mapping between the features and labels.

The architecture is a 1-d convolutional highway network. The dataset is available here: https://www.kaggle.com/datasnaek/mbti-type. If anyone 
is curious about the data preprocessing, I will gladly provide the script, it's just not pretty to look at.
'''

import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import highway_conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.normalization import batch_normalization






#Constants
LR=0.001
dropout_rate=0.8
batch_size=16
max_len=1000
lstm_dim=128
embedding_dim=128
lexicon_size=2001
num_epochs=10


features=np.load('features.npy') 
labels=np.load('labels.npy')



net = input_data(shape=[None, max_len], name='input')
net = tflearn.embedding(net, input_dim=lexicon_size, output_dim=embedding_dim)

for i in range(10):
	for j in [3, 2, 1]:
		net=highway_conv_1d(net, 16, j, activation='elu')
	net = max_pool_1d(net, 2)
	net = batch_normalization(net)


net = fully_connected(net, 128, activation='elu')
net = fully_connected(net, 256, activation='elu')
net = fully_connected(net, 16, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='target')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(features, labels, n_epoch=num_epochs, validation_set=0.1, show_metric=True, run_id='convnet_highway_mb')


