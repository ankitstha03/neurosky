# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
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
count=1
data = pd.read_csv('advantaold.csv', names=['att','med','poo','rawValue','label'])
X_train=data.rawValue.tolist()
y_train=data.label.tolist()
data2 = pd.read_csv('advantaold.csv', names=['att','med','poo','rawValue','label'])
X_test=data2.rawValue.tolist()
y_test=data.label.tolist()
y_test=np.array(y_test)
# plot 4 images as gray scale
tsts=[]
print(X_train[0])
for i in range(len(X_train)):
	# apen=[]
	
	temp=literal_eval(X_train[i])
	temp2=literal_eval(X_train[i])
	# count+=1
	# print(count)
	for j in range(len(temp)):
		for k in range(len(temp)):
			temp[j][k]=temp2[k][j]
	temp3=np.array(temp)
	tsts.append(temp3)

X_train=np.array(tsts)
print(X_train[0])
X_train = X_train.reshape(len(X_train), 8, 8)
y_train=np.array(y_train)
y_train = y_train.reshape(len(y_train), 1)

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

model = Sequential()
#model.add(Dense(16, input_dim=8, activation='relu'))

model.add(Bidirectional(LSTM(128, dropout=0.15, recurrent_dropout=0.15, return_sequences=True), input_shape=(8,8)))
model.add(Bidirectional(LSTM(32, dropout=0.15, recurrent_dropout=0.15,  return_sequences=True)))
model.add(Bidirectional(LSTM(8)))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history=model.fit(X_train, y_train, epochs=100, batch_size=150, validation_split=0.3)
# Final evaluation of the model


model_json = model.to_json()
with open("model4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model4.h5")
print("Saved model to disk")

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
 

 
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 

# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# json_file = open('model4.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model4.h5")
# print("Loaded model from disk")
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# ynew = loaded_model.predict_classes(X_train)
# for i in range(len(y_test)):
	# if(ynew[i]==y_test[i]):
		# counta+=1
# print(counta)