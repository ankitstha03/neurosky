# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

data2 = pd.read_csv('dataset4.csv', names=['att','med','poo','rawValue','label'])
X_test=data2.rawValue.tolist()
y_test=data2.label.tolist()
# plot 4 images as gray scale
tsts2=[]
for i in range(len(X_test)):
	temp=literal_eval(X_test[i])
	temp2=np.array(temp)
	tsts2.append(temp2)
X_test=np.array(tsts2)
X_test = X_test.reshape(len(X_test), 1, 8)
y_test=np.array(y_test)
y_test = y_test.reshape(len(y_test), 1)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

