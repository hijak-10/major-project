import  cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math 
import time 
import tensorflow as tf



detector= HandDetector(maxHands=2)
classifier1 = Classifier(r"C:\Users\lENOVOO\Desktop\Mproject\ModelNewNew\KKKeras.h5",r"C:\Users\lENOVOO\Desktop\Mproject\ModelNew\labels.txt")
classifier2 = Classifier(r"C:\Users\lENOVOO\Desktop\Mproject\2NewModel3\keras_model.h5",r"C:\Users\lENOVOO\Desktop\Mproject\2NewModel3\labels.txt")
cap= cv2.VideoCapture(0)
labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

counter=0
offset=20
imgSize=400

while  True:
    success , img=cap.read()
    hands, img=detector.findHands(img)
    
    
    
    if (hands):
        hand=hands[0]

        x,y,w,h= hand['bbox']

        n1hand = hand["type"]
    
        if  len(hands)==1:
        
         imgWhite=np.ones((imgSize, imgSize,3),np.uint8)*255
         imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]
        
         imgCropShape=imgCrop.shape
        
         aspectRatio=h/w
        
         if aspectRatio>1:
             k=imgSize/h
             wCal=math.ceil(k*w)
             imgResize= cv2.resize(imgCrop,(wCal,imgSize))
             imgResizeShape=imgResize.shape
             wGap=math.ceil((imgSize-wCal)/2)
             imgWhite[:,wGap:wCal+wGap]=imgResize
             prediction ,index=classifier1.getPrediction(imgWhite)
             print(prediction,index)
         else:
             k=imgSize/w
             hCal=math.ceil(k*h)
             imgResize= cv2.resize(imgCrop,(imgSize,hCal))
             imgResizeShape=imgResize.shape
             hGap=math.ceil((imgSize-hCal)/2)
             imgWhite[hGap:hCal+hGap,:]=imgResize
             prediction ,index=classifier1.getPrediction(imgWhite)
             print(prediction,index)
         cv2.imshow("Prediction ASL",imgWhite)
        if  len(hands)==2:
         

         prediction ,index=classifier2.getPrediction(img)
         print(prediction,index)
         ##cv2.putText(img,labels[index],(x+20,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
         cv2.imshow("Prediction",img)
    elif cv2.waitKey(1) == 27:
        break
    cv2.waitKey(1)

