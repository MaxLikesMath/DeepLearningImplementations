''' A 1-dimensional convolutional highway network for sentiment analysis. This script is based on the following two papers: https://arxiv.org/abs/1408.5882 and
https://arxiv.org/abs/1709.03247
'''

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import highway_conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.normalization import batch_normalization
from tflearn.datasets import imdb

LR=0.001
num_epochs=10


train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)

net = input_data(shape=[None, 100], name='input')
net = tflearn.embedding(net, input_dim=10000, output_dim=128)

for i in range(5):
	for j in [3, 2, 1]:
		net=highway_conv_1d(net, 16, j, activation='elu')
	net = max_pool_1d(net, 2)
	net = batch_normalization(net)


net = fully_connected(net, 128, activation='elu')
net = fully_connected(net, 256, activation='elu')
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='target')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=num_epochs, validation_set=(testX, testY), show_metric=True, run_id='convnet_highway_imdb')












