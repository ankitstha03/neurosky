#from keras.datasets import mnist
import matplotlib.pyplot as plt
import csv
import datetime
import pandas as pd
from ast import literal_eval
    
data = pd.read_csv('train_set.csv', names=['b','c','d','label'])
data.dropna(inplace=True)
rvvv=data.d.tolist()
b=literal_eval(rvvv[5])
a = []
c = []
# for i in range(len(b)):
	# if (i%2) is 0:
		# a.append(b[i])
	# else:
		# c.append(b[i])
# plot 4 images as gray scale
x=[0,1,2,3,4,5,6,7]

plt.plot(b)

# show the plot
plt.show()