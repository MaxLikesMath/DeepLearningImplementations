'''An implementation of the DeXpression network described in https://arxiv.org/abs/1509.05371 using tensorflow and tflearn. This 
is run on the CK+ dataset, available here: http://www.consortium.ri.cmu.edu/ckagree/. 
'''

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np


features=np.load('dex_feat.npy')
labels=np.load('dex_lab.npy')

imagesize=50
dropout_rate=0.8
LR=0.001
batch_size=128
num_epoch=10

padding='VALID'

net=input_data(shape=[None, imagesize, imagesize, 1])
net=conv_2d(net, 64, 7, strides=2, padding=padding, activation=None)
net=tf.nn.relu(net)
net=max_pool_2d(net, 3, strides=2, padding=padding)
net=tflearn.batch_normalization(net)

net_1=conv_2d(net, 96, 1, padding=padding)
net_1=tf.nn.relu(net_1)
net_2=max_pool_2d(net, 3, strides=1, padding=padding)
net_3=conv_2d(net_1, 208, 3, padding=padding)
net_3=tf.nn.relu(net_3)
net_4=conv_2d(net_2, 64, 1, padding=padding)
net_4=tf.nn.relu(net_4)
chunk_1=tflearn.merge([net_3, net_4], mode='concat', axis=3)


net_5=conv_2d(chunk_1, 96, 1, padding=padding)
net_5=tf.nn.relu(net_5)
net_6=max_pool_2d(chunk_1, 3, strides=1, padding=padding)
net_7=conv_2d(net_5, 208, 3, padding=padding)
net_7=tf.nn.relu(net_7)
net_8=conv_2d(net_6, 64, 1, padding=padding)
net_8=tf.nn.relu(net_8)
chunk_2=tflearn.merge([net_7, net_8], mode='concat', axis=3)

net=tflearn.flatten(chunk_2)
net=dropout(net, dropout_rate)
net=fully_connected(net, 7, activation='softmax')
net=regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR )

model=tflearn.DNN(net)
model.fit(features, labels, n_epoch=num_epoch, validation_set=0.1, show_metric=True, batch_size=batch_size)














