# This file is to split the folders into train,test,val
import os 
os.chdir(r'E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\data2')

import splitfolders

#splitfolders.ratio("Indian",output="output",seed=1337,ratio=(.8, 0.1,0.1))

splitfolders.fixed("Inv",output="output",seed=1337,fixed=(100,100))