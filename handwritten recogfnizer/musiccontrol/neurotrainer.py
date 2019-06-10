# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras import optimizers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from ast import literal_eval

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
counta=0
data = pd.read_csv('train_set.csv', names=['att','med','poo','rawValue','label'])
X_train=data.rawValue.tolist()
y_train=data.label.tolist()
data2 = pd.read_csv('train_set.csv', names=['att','med','poo','rawValue','label'])
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
X_train = X_train.reshape(len(X_train), 8, 256)
y_train=np.array(y_train)
y_train = y_train.reshape(len(y_train), 1)

y_train = np_utils.to_categorical(y_train)
accu=[]
valac=[]
loos=[]
valloos=[]
batch=[]
cnt=0.0001
num_classes = y_train.shape[1]
history=[]
for i in range(5): 
	model = Sequential()
	# model.add(Conv1D(filters=256, input_shape=(1,2048), kernel_size=1, padding='same', activation='relu'))
	# model.add(MaxPooling1D(pool_size=1))
	model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True) ,input_shape=(8,256)))
	model.add(Bidirectional(LSTM(32, dropout=0.25, recurrent_dropout=0.25,return_sequences=True)))
	model.add(Bidirectional(LSTM(8, dropout=0.25, recurrent_dropout=0.25)))
	model.add(Dropout(0.25))
	# model.add(LSTM(32))
	model.add(Dense(num_classes, activation='softmax'))
	adm = optimizers.adam(lr=cnt)
	model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
	print(model.summary())
	temp=model.fit(X_train, y_train, epochs=40, batch_size=70, validation_split=0.2)
	history.append(temp)
	cnt=cnt*10
	# accu.append(history.history['acc'][36])
	# valac.append(history.history['val_acc'][36])
	# loos.append(history.history['loss'][36])
	# valloos.append(history.history['val_loss'][36])
# Final evaluation of the model


model_json = model.to_json()
with open("model0.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model0.h5")
print("Saved model to disk")
 
print(history[0].history.keys())
#  "Accuracy"
plt.plot(history[0].history['acc'], 'r--')
plt.plot(history[1].history['acc'], 'b-+')
plt.plot(history[2].history['acc'], 'g-.')
plt.plot(history[3].history['acc'], 'y:')
plt.plot(history[4].history['acc'], 'c-')
plt.title('Effect of learning rate on accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['0.0001', '0.001', '0.01', '0.1', '1'], loc='upper left')
plt.show()
# "Loss"
# plt.plot( batch, loos, 'r--')
# plt.plot( batch, valloos)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('batch size')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
plt.plot(history[0].history['loss'], 'r--')
plt.plot(history[1].history['loss'], 'b-+')
plt.plot(history[2].history['loss'], 'g-.')
plt.plot(history[3].history['loss'], 'y:')
plt.plot(history[4].history['loss'], 'c-')
plt.title('Effect of learning rate on loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['0.0001', '0.001', '0.01', '0.1', '1'], loc='upper right')
plt.show()

 
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 

# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# json_file = open('model3.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model3.h5")
# print("Loaded model from disk")
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# ynew = loaded_model.predict(X_train)
# ynew2=np.argmax(ynew,axis=1)
# for i in range(len(y_test)):
	# if(ynew2[i]==y_test[i]):
		# counta+=1
# print(counta)
