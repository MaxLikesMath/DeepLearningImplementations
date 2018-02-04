# Deep Learning Implementations
This is a repository of implementations of deep learning papers, along with personal projects and experiments. Most are written
using a mix of TFLearn and Tensorflow. 

# Project Descriptions
Basic descriptions of the projects. They are all commented with the relevant information, so if you're curious about anything, check there. If it's not there, feel free to ask me!

DeXpression- Implementation of the DeXpression paper: https://arxiv.org/abs/1509.05371
	
Scout Net- A network architecture based off of DeXpression that utilizes highway convolutional layers to classify facial expressions from the CK+ dataset.
	
IMDB Highway- A network for sentiment analysis that uses 1-d convolutional highway layers.

GRU Cart- A network that utilizes gated recurrent units to solve OpenAI's cartpole environment. Trains in about 3 seconds.

MB Exp- An experiment on the Myers-Briggs dataset from Kaggle that attempts to classify people based on Twitter posts.

Hate Speech- Using a recurrent network with GRU cells to classify hate speech.

Attention Mech- A function made in Tensorflow that replicates the attention mechanism described in http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

seq2seq- A basic sequence2sequence function using LSTM cells without attention that's based off of the Tensorflow tutorial with some minor changes and work arounds for issues with Deepcopy.
