import  cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time 

detector= HandDetector(maxHands=1)
nhand=detector.mpHands
cap= cv2.VideoCapture(0)

folder = r"C:\Users\lENOVOO\Desktop\OG Data\ASL Y"
counter=0
offset=20
imgSize=400
while  True:
    success , img=cap.read()
    hands , img=detector.findHands(img)
    
    if hands:
        hand=hands[0]
        x,y,w,h= hand['bbox']
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
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize= cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize


        cv2.imshow("white image",imgWhite)

    cv2.imshow("Image",img)
    key=cv2.waitKey(1)

    if key==ord("s"):
        counter+=1
        kk=str(counter)
        cv2.imwrite(folder+'/Image_'+kk+'.jpg',imgWhite)
        print(counter)