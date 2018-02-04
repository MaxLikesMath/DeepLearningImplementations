import tensorflow as tf
import numpy as np
import sys

class seq2seq(object):
	setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self) #Because of a threading error with Tensorflow, this line is needed as a work around
	def __init__(self, encode_seq_len, decode_seq_len, encoder_vocab_size, decoder_vocab_size, embedding_dim, num_layers, ckpt_path, num_epochs, learning_rate=0.001,  model_name='seq2seq_model'):
		
		
		self.encode_seq_len = encode_seq_len
		self.decode_seq_len = decode_seq_len
		self.ckpt_path = ckpt_path
		self.num_epochs = num_epochs
		self.model_name = model_name

		def comp_graph(): #We need to create our computation graph.
			tf.reset_default_graph()
			
			#A seq2seq model is effectively an encoder-decoder architecture that's built for sequence data.
			#This model is a basic LSTM architecture without an explicit attention mechanism, but one can be added without much struggle.
			
			self.encoder_inputs=[ tf.placeholder(shape=[None,], dtype=tf.int64, name='ei_{}'.format(t)) for t in range(encode_seq_len) ] 
				
			self.labels = [ tf.placeholder(shape=[None,], dtype=tf.int64, name='ei_{}'.format(t)) for t in range(decode_seq_len) ]
			self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]
				
			self.dropout_rate = tf.placeholder(tf.float32) #Dropout is a powerful normalizing tool for NLP, so it's a good idea to include it.
			basic_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(embedding_dim, state_is_tuple=True), output_keep_prob=self.dropout_rate)
			
			#You can use different recurrent cells like GRUs.
			
			stacked_LSTM = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True) #Let's us determine depth.
				
			with tf.variable_scope('decoder') as scope:
				
				self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs, stacked_LSTM, encoder_vocab_size, decoder_vocab_size, embedding_dim)
				scope.reuse_variables()
				self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs, stacked_LSTM, encoder_vocab_size, decoder_vocab_size, embedding_dim,feed_previous=True)
				#We use legacy_seq2seq here which is a deprecated library in Tensorflow, but it works well for our purposes.
			
			
			loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
			self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, decoder_vocab_size)
			self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
			#The Adam optimizer is standard in NLP, though SGD with momentum can be used effectively in its place.
			
		
		comp_graph() #Create our graph.
		
		
		#The following are basic helper functions to run the seq2seq model.
		
	def Get_dict(self, X, Y, dropout_rate):
		feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.encode_seq_len)}	
		feed_dict.update({self.labels[t]: Y[t] for t in range(self.decode_seq_len)})
		feed_dict[self.dropout_rate] = dropout_rate
			
		return feed_dict
			
			
	def train_batch(self, sess, train_batch_gen):
		batchX, batchY = train_batch_gen.__next__()
		feed_dict = self.Get_dict(batchX, batchY, dropout_rate=0.2) #keep prob
		_, loss_v = sess.run([self.train_op, self.loss], feed_dict)
		return loss_v
			
	def eval_step(self, sess, eval_batch_gen):
		batchX, batchY = eval_batch_gen.__next__()
		feed_dict = self.Get_dict(batchX, batchY, dropout_rate=1.)
		loss_v, decoder_outputs_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
		decoder_outputs_v = np.array(decoder_outputs_v).transpose([1,0,2])
		return loss_v, decoder_outputs_v, batchX, batchY

	def eval_batches(self, sess, eval_batch_gen, num_batches):
		losses = []
		for i in range(num_batches):
			loss_v, decoder_outputs_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
			losses.append(loss_v)
		return np.mean(losses)

	def train(self, train_set, valid_set, sess=None, save=True):
		saver = tf.train.Saver()
			
		if not sess:
			sess = tf.Session()
			sess.run(tf.global_variables_initializer())
				
		for i in range(self.num_epochs):
			try:
				self.train_batch(sess, train_set)
					
				if i%2==0:
					if save==True:
						saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
						
					val_loss = self.eval_batches(sess, valid_set, 16)
					print('val   loss : {0:.6f}'.format(val_loss))
						
					sys.stdout.flush()
			except KeyboardInterrupt:
				print('Interrupted by user at iteration {}'.format(i))
				self.session = sess
				return sess
				
	def restore_last_session(self):
		saver = tf.train.Saver()
				
		sess = tf.Session()
				
		ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		return sess


	def predict(self, sess, X):
		feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.encode_seq_len)}
		feed_dict[self.dropout_rate] = 1.
		decoder_outputs_v = sess.run(self.decode_outputs_test, feed_dict)
		decoder_outputs_v = np.array(decoder_outputs_v).transpose([1,0,2])

		return np.argmax(decoder_outputs_v, axis=2)
				
				
				
				

			
			
			
