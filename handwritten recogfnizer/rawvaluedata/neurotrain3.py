import numpy as np
from attention_net import Attention
from keras.models import model_from_json
from keras.utils import plot_model
import pandas as pd
from ast import literal_eval
from subprocess import call
import webbrowser
from time import sleep


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(
    loaded_model_json, custom_objects={"Attention":Attention})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
webbrowser.open('http://169.254.184.158:8000/iot/home', new=2)

while(True):

	exit_code = call("python2 nskydatapoor.py", shell=True)
	csvfiles = [
		'datasetexp.csv'
	]

	for f in range(len(csvfiles)):
		counter = 0
		checkData = pd.read_csv(
			csvfiles[f], names=['att', 'med', 'poo', 'rawValue', 'label'])
		X = checkData.rawValue.tolist()
		y = checkData.label.tolist()
		y = np.array(y)
		# plot 4 images as gray scale
		test_data = []
		for i in range(len(X)):
			temp = np.array(literal_eval(X[i]))
			test_data.append(temp)

		X = np.array(test_data)
		X = X.reshape(len(X), 2048, 1)

		loaded_model.compile(
			loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])
		y_new = loaded_model.predict_classes(X)
		for i in range(len(y)):
			if (y_new[i] == y[i]):
				counter += 1
		print(y_new)
		if(y_new==0):
			webbrowser.open('http://169.254.184.158:8000/iot/asd/1')
		if(y_new==1):
			webbrowser.open('http://169.254.184.158:8000/iot/asd/2')
		if(y_new==2):
			webbrowser.open('http://169.254.184.158:8000/iot/asd/3')
		if(y_new==3):
			webbrowser.open('http://169.254.184.158:8000/iot/asd/4')
	
	sleep(5)
			

	

