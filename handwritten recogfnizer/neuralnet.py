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

data2 = pd.read_csv('dataset4.csv', names=['att','med','poo','rawValue','label'])
X_test=data2.rawValue.tolist()
# plot 4 images as gray scale



json_file = open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model3.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
sleep(1)
print("taking input started")
object1=NeuroPy("COM15")
count=0
count0=0
count1=0
val=0
cnter=0
qwe=0
arr=[0.000]*2048
def attention_callback(attention_value):
	if(object1.attention>=15 and object1.meditation>=15 and object1.poorSignal==0 and attention_value<30000):
		global cnter
		cnter+=1
		for i in range(len(arr)-1):
			arr[i]=arr[i+1]
		arr[2047]=attention_value
		if(cnter==400 ):
			cnter=0
			temp2=[arr]
			temp2=np.array(temp2)
			temp2 = temp2.reshape(len(temp2), 1, 2048)
			ynew = loaded_model.predict_proba(temp2)
			print("X=%s, Predicted=%s" % (temp2, ynew))
			global count
			global count0
			global count1
			count+=1
			# if(ynew[0][0]>0.7):
				# print("left")
				# count0+=1
			# if(ynew[0][1]>0.7):
				# print("right")
				# count1+=1
			# if(count>=5):
				# count=0
				# if(count0>=3):
					# print("left")
					# return 30
					# count0=0
					# count1=0
				# elif(count1>=3):
					# print("right")
					# count0=0
					# return 40
					# count1=0
				# else:
					# print("null")
					# count0=0
					# count1=0

		#with open('dataset3.csv', 'ab') as f:
			#writer = csv.writer(f)
			#writer.writerow([object1.attention, object1.meditation, object1.poorSignal, temp, 1])
			
	return None 

#set call back: 
object1.setCallBack("rawValue",attention_callback) 
#call start method 
object1.start()
while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print("geda meditationg too much")
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset