from NeuroPy import NeuroPy
from time import sleep
import csv

val=0
cnter=0
qwe=0
arr=[0.000]*1024
object1=NeuroPy("COM15")
with open('dataset5.csv', 'wb') as f:
	writer = csv.writer(f)
			
def attention_callback(attention_value): 
	if(object1.attention>=15 and object1.meditation>=15 and object1.poorSignal==0):
		global cnter
		cnter+=1
		for i in range(len(arr)-1):
			arr[i]=arr[i+1]
		arr[1023]=attention_value
		if(cnter==1024):
			global qwe
			global val
			if(qwe==0):
				qwe=1
				val=input("(0 for left and 1 for right and 2 for ntg)")
				cnter=0
				return None
			with open('dataset5.csv', 'ab') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			val=input("(0 for left and 1 for right and 2 for ntg)")
			cnter=0
	return None 

#set call back: 
object1.setCallBack("rawValue",attention_callback) 
#call start method 
val=input("(0 for left and 1 for right and 2 for ntg)")
object1.start() 

while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print("geda meditationg too much")
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset