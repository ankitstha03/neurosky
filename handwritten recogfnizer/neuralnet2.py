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
from graphics import *
import csv
class neural():
	data2 = pd.read_csv('dataset4.csv', names=['att','med','poo','rawValue','label'])
	X_test=data2.rawValue.tolist()
	# plot 4 images as gray scale
	tsts2=[]
	for i in range(len(X_test)):
		temp=literal_eval(X_test[i])
		temp2=np.array(temp)
		tsts2.append(temp2)
	X_test=np.array(tsts2)
	X_test = X_test.reshape(len(X_test), 1, 8)


	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	sleep(1)
	print("taking input started")
	object1=NeuroPy("COM11")


	def attention_callback(attention_value):
		win = GraphWin('Face', 200, 150) # give title and dimensions
		win.yUp() # make right side up coordinates!
		
		head = Circle(Point(40,100), 25) # set center and radius
		head.setFill("yellow")
		head.draw(win)

		if(object1.attention>=5 and object1.meditation>=5):
			print(object1.poorSignal)
			temp=[object1.delta, object1.theta, object1.lowAlpha, object1.highAlpha, object1.lowBeta, object1.highBeta, object1.lowGamma, object1.midGamma]
			#with open('dataset3.csv', 'ab') as f:
				#writer = csv.writer(f)
				#writer.writerow([object1.attention, object1.meditation, object1.poorSignal, temp, 0])
			temp2=[temp]
			temp2=np.array(temp2)
			temp2 = temp2.reshape(len(temp2), 1, 8)
			ynew = loaded_model.predict_proba(temp2)
			print("X=%s, Predicted=%s" % (temp2, ynew))
			if(ynew[0][0]>0.69):
				print("left")
				eye1 = Circle(Point(30, 105), 5)
				eye1.setFill('blue')
				eye1.draw(win)
			if(ynew[0][1]>0.69):
				print("right")
				eye2 = Line(Point(45, 105), Point(55, 105)) # set endpoints
				eye2.setWidth(3)
				eye2.draw(win)
		
		return None 

	#set call back: 
	object1.setCallBack("delta",attention_callback) 
	#call start method 
	object1.start() 

	while True:
		if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
			print("geda meditationg too much")
			#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset