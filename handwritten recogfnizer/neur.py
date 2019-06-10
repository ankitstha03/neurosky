import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
import pandas as pd
from keras.utils import np_utils
from ast import literal_eval
from NeuroPy import NeuroPy
from time import sleep
import csv


json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data = pd.read_csv('dataset3.csv', names=['att','med','poo','rawValue','label'])
X_train=data.rawValue.tolist()
ssd=data.label.tolist()

# plot 4 images as gray scale
tsts=[]
for i in range(len(X_train)):
	temp=literal_eval(X_train[i])
	temp2=np.array(temp)
	tsts.append(temp2)

X_train=np.array(tsts)
X_train = X_train.reshape(len(X_train), 1, 8)
y_train=np.array(ssd)
y_train = y_train.reshape(len(y_train), 1)

delet=[]
for i in range(len(X_train)):
	dsa=[]
	dsa=[X_train[i]]
	temp2=np.array(dsa)
	temp2 = temp2.reshape(len(temp2), 1, 8)
	ynew = loaded_model.predict_classes(temp2)
	print(ynew)
	if(ynew[0]!=ssd[i]):
		delet.append(i)
data=data.drop(delet)
data.to_csv('dataset4.csv')