import  cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time 

detector= HandDetector(maxHands=2)
nhand=detector.mpHands
cap= cv2.VideoCapture(0)
folder = r"C:\Users\lENOVOO\Desktop\2OG data\BSL A"
counter=0
offset=20
imgSize=400
while  True:
    success , img=cap.read()
    hands , img=detector.findHands(img)
    
    if hands:
        hand=hands[0]
        x,y,w,h= hand['bbox']

    cv2.imshow("Image",img)
    key=cv2.waitKey(1)

    if key==ord("s"):
        counter+=1
        kk=str(counter)
        cv2.imwrite(folder+'/Image_'+kk+'.jpg',img)
        print(counter)
    elif cv2.waitKey(1) == 27:
        break