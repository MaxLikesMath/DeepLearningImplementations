''' This is a GRU net trained to detect hate speech. The dataset comes from https://github.com/t-davidson/hate-speech-and-offensive-language.
After 5 epochs this net achieves about ~90% accuracy on the validation set.

'''

import numpy as np
import pandas as pd
import tflearn
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
import re
from sklearn.preprocessing import LabelBinarizer



LR=0.001
dropout_rate=0.8
batch_size=128
max_len=150
lstm_dim=128
embedding_dim=128
num_epoch=2




Stemmer=nltk.stem.PorterStemmer() #
stop_words=set(stopwords.words('english')) #We import the "stopwords" corpus from nltk so we can simplify our lexicon down the line.

#We start by preprocessing our data.
data=pd.read_csv('C:\Python Practice\labeled_data.csv', index_col='class')

#We also want to clean up this data. We do this by cleaning up our posts section of our data.
posts=data.tweet.tolist() #This gives us our data to manipulate


link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
for post in posts:
	links = re.findall(link_regex, post)
	for link in links:
		post = post.replace(link[0], ', ')

unwanted_stuff=['|||','(',')',',',':','.',';','"', '\\', '-', '=', '!', '&', '_','1','2','3','4','5','6','7','8','9','0', '?', "'"] #This a list of the symbols we don't want in our data.
i=0
for post in posts:	#This for loop converts all the text to lowercase, removes unwanted stuff, tokenizes, stems, then returns the post.
	post=post.lower()
	re.sub("[^a-zA-Z]+", "", post)
	for stuff in unwanted_stuff:
		post=post.replace(stuff,"")
	Stemmer.stem(post)
	post=word_tokenize(post)
	posts[i]=post
	i=i+1


for post in posts:
	for words in post:
		if words[0] in ['@', '#']:
			post.remove(words)
		else:
			for word in stop_words:
					if words==word:
						post.remove(words)
						break




lexicon=[]
lexicon_as_int=[] #We also want to store our lexicon as a set of integers.
j=0
for post in posts:
	for n in post:
		if n not in lexicon:
			lexicon.append(n)
			lexicon_as_int.append(j)
			j=j+1
		else:
			break
lexicon_size=len(lexicon_as_int)
print(lexicon_size)

#Turn our posts into vectors.
posts_as_int=[]
for post in posts:
	post_int=[]
	for n in post:
		t=0
		for k in lexicon:
			if n==k:
				post_int.append(lexicon_as_int[t])
				break
			else:
				t=t+1
	posts_as_int.append(post_int)
	
#Now we make our feature set.
features=tflearn.data_utils.pad_sequences(posts_as_int, maxlen=150)
features=np.array(features)
np.save('Hate_Feat', features)

#We also need labels.
labels=data.index.tolist()
convert_to_one_hot=LabelBinarizer()
labels=convert_to_one_hot.fit_transform(labels)
labels=np.array(labels)

np.save('Hate_Lab', labels)



net = tflearn.input_data([None, 150])
net = tflearn.embedding(net, input_dim=lexicon_size, output_dim=256)
net = tflearn.gru(net, 256, dropout=dropout_rate, return_seq=True)
net = tflearn.gru(net, lstm_dim)
net = tflearn.fully_connected(net, 64, activation='relu')
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')


model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(features, labels, n_epoch=num_epoch, validation_set=0.1, show_metric=True, batch_size=batch_size)








