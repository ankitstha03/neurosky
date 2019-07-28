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
from keras.models import load_model
from keras.utils import np_utils
from ast import literal_eval
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
count0=0
count1=0
count2=0
count3=0

data2048 = pd.read_csv('redyellowtestseen.csv', names=['att','med','poo','rawValue','label'])
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
yy=np.array(y_train2048)
y_train2048 = y_train2048.reshape(len(y_train2048), 1)
y_train2048 = np_utils.to_categorical(y_train2048)
num_classes2048 = y_train2048.shape[1]



model = load_model('model5032.h5', custom_objects={"Attention":Attention})
print("Loaded model from disk")
ynew = model.predict_classes(X_train2048)
for i in range(len(yy)):
    if(ynew[i]==yy[i]):
        if(ynew[i]==0):
            count0+=1
        elif(ynew[i]==1):
            count1+=1
        elif(ynew[i]==2):
            count2+=1
        elif(ynew[i]==3):
            count3+=1
print(count0)
print(count1)
print(count2)
print(count3)
# # ynew2=np.argmax(ynew,axis=1)
# for i in range(len(y_test)):
	# if(ynew2[i]==y_test[i]):
		# 
#print(counta)