# Some changes made therefore again copy paste from vs code

import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import cv2
import sys, os
import matplotlib.pyplot as plt
from string import ascii_uppercase



print("Enter the model number to run")
print("1.CNN (1 conv layers) 2.CNN (2 conv layers) 3.CNN (3 conv layers) 4.ANN 5.VGG19 6.InceptionV3 7.Resnet50")
temp='3'
print("algorithm number selected is {}".format(temp))

#path 

if temp == '1':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN1(2)")
    jsmodel="model-bw1.json"
    h5model='model-bw1.h5'
if temp=='2':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN2(2)")
    jsmodel="model-bw2.json"
    h5model='model-bw2.h5'
if temp=='3':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN3(2)")
    jsmodel="model-bw32.json"
    h5model='model-bw32.h5'
if temp=='4':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\ANN(2)")
    jsmodel="model-bwA.json"
    h5model='model-bwA.h5'
if temp=='5':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\VGG19_2")
    jsmodel="model-bwVGG192.json"
    h5model="model-bwVGG192.h5"
if temp=='6':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\InceptionV3_2")
    jsmodel="model-bwInceptionV32.json"
    h5model='model-bwInceptionV32.h5'
if temp=='7':
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\Resnet502")
    jsmodel="model-bwResnet.json"
    h5model='model-bwResnet.h5'




# Loading the model
json_file = open(jsmodel, "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights(h5model)
print("Loaded model from disk")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame,1)  # 1 tha   (0 karne se img vertically flip ho ja ra h)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
 
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
   
    # Resizing the ROI so it can be fed to the model for prediction
    if temp=='5' or temp=='6' or temp=='7':
        roi = cv2.resize(roi, (200, 150))
    else:
        roi = cv2.resize(roi, (200, 200))    #200,150
#    roi = cv2.resize(roi, (64, 64))
    

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
    kernel = np.ones((3,3),np.uint8)   # later on
         
    # define range of skin color in HSV   (later on)
    lower_skin = np.array([20,40,40], dtype=np.uint8)  #0,20,70   (20,40,50)-Good
    upper_skin = np.array([218,231,250], dtype=np.uint8) # 20,255,255
        
     #extract skin colur image    # later on
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    #cv2.imshow("mask",mask)
        
    #extrapolate the hand to fill dark spots within  #later on
    mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    #cv2.imshow("Gausmask",mask)
    
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    #cv2.imshow("mask",mask)
    
    th3 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #mask/blur
    
    #cv2.imshow("TH3",th3)
    
    ret, test_image = cv2.threshold(mask, 70 , 255,cv2.THRESH_OTSU) #th3 '''
    
    # Resizing the ROI so it can be fed to the model for prediction
    if temp=='5' or temp=='6' or temp=='7':
        roi = cv2.resize(roi, (200, 150))
    else:
        roi = cv2.resize(roi, (200, 200))


                                                            #grayscalekarega
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)      ######
    # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)   #####
    
    
    ''' 
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
     
    _, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)'''
    
    
    cv2.imshow("test", test_image)
    # Batch of 1
    if temp=='5' or temp=='6' or temp=='7':
        result = loaded_model.predict(test_image.reshape(1,100,100, 3))    #3 only for TL as there RGB
    else:
        result = loaded_model.predict(test_image.reshape(1,200,200, 1)) # not TL
    prediction={}                                      
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
