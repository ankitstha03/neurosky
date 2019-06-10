from NeuroPy import NeuroPy
from time import sleep
import threading
import csv

class Nsky():
    def __init__(self):
        self.value=0
        self.counter=0
        self.skip_value=0
        self.data_array =[0.000]*1024
        self.neuro_object = NeuroPy("COM15")
        self.caller()
                    
    def attention_callback(self, attention_value): 
        if(self.neuro_object.attention>=15 and self.neuro_object.meditation>=15 and self.neuro_object.poorSignal==0):
            self.counter+=1
            for i in range(len(self.data_array)-1):
                self.data_array[i]=self.data_array[i+1]
            self.data_array[1023]=attention_value
            if(self.counter==1024):
                if(self.skip_value==0):
                    self.skip_value=1
                    self.value=input("(0 for left and 1 for right and 2 for ntg)")
                    self.counter=0
                    if self.value == 3:
                        self.neuro_object.stop()
                    return None
                with open('traindata.csv', 'ab') as datafile:
                    writer = csv.writer(datafile)
                    writer.writerow([self.neuro_object.attention, self.neuro_object.meditation, self.neuro_object.poorSignal, self.data_array, self.value])
                self.value=input("(0 for left and 1 for right and 2 for ntg)")
                self.counter=0
        return None 

    def caller(self):
        #set call back: 
        self.neuro_object.setCallBack("rawValue",self.attention_callback) 
        #call start method 
        self.value=input("(0 for left and 1 for right and 2 for ntg)")
        self.neuro_object.start() 

        while True:
            if(self.neuro_object.meditation>100): #another way of accessing data provided by headset (1st being call backs) 
                print("geda meditationg too much")
                #object1.stop() #if meditation level reaches above 70, stop fetching data from the headset

if __name__ == "__main__":
   app = Nsky()