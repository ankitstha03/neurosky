# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras import regularizers
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from attention_net import Attention
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from ast import literal_eval
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
counta=0

data2048 = pd.read_csv('dataset503.csv', names=['att','med','poo','rawValue','label'])
#dataset2048 = data2048.sample(frac=1)
X_train2048=data2048.rawValue.tolist()
y_train2048=data2048.label.tolist()


tsts=[]
for i in range(len(X_train2048)):
	temp=literal_eval(X_train2048[i])
	temp2=np.array(temp)
	tsts.append(temp2)

X_train2048=np.array(tsts)
X_train2048 = X_train2048.reshape(len(X_train2048), 256, 8)
y_train2048=np.array(y_train2048)
y_train2048 = y_train2048.reshape(len(y_train2048), 1)
y_train2048 = np_utils.to_categorical(y_train2048)
num_classes2048 = y_train2048.shape[1]



model2048 = Sequential()
model2048.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True ,input_shape=(256,8)))
model2048.add(Dropout(0.2))
model2048.add(LeakyReLU(alpha=0.02))
model2048.add(LSTM(64, dropout=0.1, activation='tanh', recurrent_dropout=0.1,return_sequences=True))
model2048.add(Attention())
model2048.add(Dense(num_classes2048, activation='softmax'))
model2048.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2048.summary())
history2048=model2048.fit(X_train2048, y_train2048, epochs=63, batch_size=100, shuffle=True)

model2048.save('model5032.h5')

print("Saved 2048 model to disk")


print(history2048.history.keys())
#  "Accuracy"
plt.plot(history2048.history['acc'], 'r--')
plt.plot(history2048.history['val_acc'])
plt.title('model accuracy 2048')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history2048.history['loss'], 'r--')
plt.plot(history2048.history['val_loss'])
plt.title('model loss 2048')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

 