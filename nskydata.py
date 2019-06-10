from NeuroPy import NeuroPy
from time import sleep
import csv
from subprocess import call
import sys

val=0
cnter=0
qwe=0
arr=[0.000]*2048
object1=NeuroPy("COM22")
# with open('dataset5.csv', 'wb') as f:
	# writer = csv.writer(f)
print("started taking data")
def attention_callback(attention_value):
	if(object1.attention>=15 and object1.meditation>=15 and object1.poorSignal==0 and attention_value<30000):
		global cnter
		cnter+=1
		for i in range(len(arr)-1):
			arr[i]=arr[i+1]
		arr[2047]=attention_value
		if(cnter==2048):
			global qwe
			global val
			# if(qwe==0):
				# qwe=1
				# val=input("(0 for red 1 for green 2 for black and 3 for yellow)")
				# cnter=0
				# return None
			with open('datasetexp.csv', 'w') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			print("data taking completed")
			cnter=0
			sys.exit()
			
			
	return None 
with open('datasetexp.csv', 'w') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
#set call back: 
object1.setCallBack("rawValue",attention_callback) 
#call start method 
object1.start() 
sys.exit()

while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print("geda meditationg too much")
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset