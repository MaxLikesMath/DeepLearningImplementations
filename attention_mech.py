import tensorflow as tf
import numpy as np

def attention(inputs, input_size, attention_size, bi_rnn=False):
	#This is based on http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
	if bi_rnn=True:
		inputs=tf.concat(inputs, 2)
	#Our first step is to pass our weights*inputs+bias through the tanh function. We start by initializing our values:
	W=tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.2))
	b=tf.Variable(tf.random_normal([attention_size], stddev=0.2))
	u=tf.Variable(tf.random_normal([attention_size], stddev=0.2)) #This is our "context" vector.
	
	hid_rep=tf.tensordot(inputs, W, axes=1) + b #We run our input through a feed forward neural net
	hid_rep=tf.tanh(hid_rep)
	
	word_dif=tf.tensordot(hid_rep, u, axes=1) #Then we calculate the word difference.
	alpha=tf.nn.softmax(word_dif)
	output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
	
	return output




















