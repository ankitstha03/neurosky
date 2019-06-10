# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from attention import Attention
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
counta=0
data = pd.read_csv('unseen0.csv', names=['att','med','poo','rawValue','label'])
X_train=data.rawValue.tolist()
y_train=data.label.tolist()
data2 = pd.read_csv('unseen0.csv', names=['att','med','poo','rawValue','label'])
X_test=data2.rawValue.tolist()
y_test=data.label.tolist()
y_test=np.array(y_test)
# plot 4 images as gray scale
tsts=[]
for i in range(len(X_train)):
	temp=literal_eval(X_train[i])
	temp2=np.array(temp)
	tsts.append(temp2)

X_train=np.array(tsts)
X_train = X_train.reshape(len(X_train), 2048, 1)
y_train=np.array(y_train)
y_train = y_train.reshape(len(y_train), 1)

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

# model = Sequential()
# model.add(Conv1D(filters=256, input_shape=(1,2048), kernel_size=1, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=1))
# model.add(LSTM(256, return_sequences=True))
# model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
# model.add(LSTM(32))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=50, batch_size=64)
# # Final evaluation of the model


# model_json = model.to_json()
# with open("model3.json", "w") as json_file:
    # json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model3.h5")
# print("Saved model to disk")
 
# later...
 
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 

# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'Attention': Attention})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ynew = loaded_model.predict(X_train)
ynew2=np.argmax(ynew,axis=1)
for i in range(len(y_test)):
	if(ynew2[i]==y_test[i]):
		counta+=1
print(counta)
