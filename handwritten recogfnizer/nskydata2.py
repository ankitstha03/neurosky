from NeuroPy import NeuroPy
from time import sleep
import csv

val=0
cnter=0
qwe=0
arrdelta=[0.000]*8
arrtheta=[0.000]*8
arrloalpha=[0.000]*8
arrhialpha=[0.000]*8
arrlobeta=[0.000]*8
arrhibeta=[0.000]*8
arrlogamma=[0.000]*8
arrmidgamma=[0.000]*8
object1=NeuroPy("COM15")
# with open('advandata.csv', 'wb') as f:
	# writer = csv.writer(f)
			
def attention_callback(attention_value): 
	if(object1.attention>=15 and object1.meditation>=15 and object1.poorSignal==0):
		global cnter
		cnter+=1
		for i in range(len(arrdelta)-1):
			arrdelta[i]=arrdelta[i+1]
			arrtheta[i]=arrtheta[i+1]
			arrloalpha[i]=arrloalpha[i+1]
			arrhialpha[i]=arrhialpha[i+1]
			arrlobeta[i]=arrlobeta[i+1]
			arrhibeta[i]=arrhibeta[i+1]
			arrlogamma[i]=arrlogamma[i+1]
			arrmidgamma[i]=arrmidgamma[i+1]
		arrdelta[7]=object1.delta
		arrtheta[7]=object1.theta
		arrloalpha[7]=object1.lowAlpha
		arrhialpha[7]=object1.highAlpha
		arrlobeta[7]=object1.lowBeta
		arrhibeta[7]=object1.highBeta
		arrlogamma[7]=object1.lowGamma
		arrmidgamma[7]=object1.midGamma
		if(cnter==8):
			global qwe
			global val
			arr=[arrdelta, arrtheta, arrloalpha, arrhialpha, arrlobeta, arrhibeta, arrlogamma, arrmidgamma]
			if(qwe==0):
				qwe=1
				val=input("(0 for left and 1 for right and 2 for ntg)")
				cnter=0
				return None
			with open('advanta.csv', 'ab') as f:
				writer = csv.writer(f)
				writer.writerow([object1.attention, object1.meditation, object1.poorSignal, arr, val])
			val=input("(0 for left and 1 for right and 2 for ntg)")
			cnter=0
	return None 

#set call back: 
object1.setCallBack("midGamma",attention_callback) 
#call start method 
val=input("(0 for left and 1 for right and 2 for ntg)")
object1.start() 

while True:
	if(object1.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
		print("geda meditationg too much")
		#object1.stop() #if meditation level reaches above 70, stop fetching data from the headset