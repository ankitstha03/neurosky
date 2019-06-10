import numpy as np
from attention_net import Attention
from keras.models import model_from_json
from keras.utils import plot_model
import pandas as pd
from ast import literal_eval
from subprocess import call

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(
    loaded_model_json, custom_objects={"Attention":Attention})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


exit_code = call("C:\Python27\python nskydata.py", shell=True)
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

	with open('datasetlab.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow([y_new])

