from NeuroPy import NeuroPy
from time import sleep

object1=NeuroPy("COM15")
def attention_callback(attention_value): 
	print("blinkval", object1.blinkStrength)
	#do other stuff (fire a rocket), based on the obtained value of attention_value 
	#do some more stuff 
	return None 

	

#set call back: 
object1.setCallBack("attention",attention_callback) 

#call start method 
object1.start() 

while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print("geda meditationg too much")
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset