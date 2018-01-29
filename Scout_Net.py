'''This is a neural network inspired by the DeXpression architecture (https://arxiv.org/abs/1509.05371) that utilizes 
a convolutional highway network (https://arxiv.org/abs/1709.03247) to speed up computation. From my initial testing, it achieves a rounded average of ~98% but
I have seen that it does very greatly, from >99% to ~96%, hinting towards the merge of the layers in DeXpression being key to its performance.
'''


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import highway_conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
import numpy as np





LR=0.001
num_epochs=20
im_size=50
batch_size=64
dropout_rate=0.8

#In order to get the dataset, one must visit http://www.consortium.ri.cmu.edu/ckagree/. 
features=np.load('dex_feat.npy')
labels=np.load('dex_lab.npy')




net=input_data(shape=[None, im_size, im_size, 1])

for i in range(5):
	for j in [3, 2, 1]:
		net=highway_conv_2d(net, 32, j, activation='elu')
	net = max_pool_2d(net, 2)
	net = batch_normalization(net)

net = fully_connected(net, 128, activation='relu')
net = dropout(net, dropout_rate)
net = fully_connected(net, 256, activation='relu')
net = dropout(net, dropout_rate)
net = fully_connected(net, 7, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='target')


model=tflearn.DNN(net)
model.fit(features, labels, n_epoch=num_epochs, validation_set=0.1, show_metric=True, batch_size=batch_size)

#If one wanted to make predictions on unlabelled data, bolt on a model.predict.

















