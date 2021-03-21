import cv2
import numpy as np
import utils


webcam=False
path='img.jpeg'
cap=cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale=3
wP=210*scale
hP=297*scale
while True:
    if webcam:success,img=cap.read()
    else: img=cv2.imread(path)
    img,conts=utils.getContours(img,draw=True,minArea=50000,filter=4)
    if len(conts)!=0:
        biggest=conts[0][2]
        # print (biggest)
        imgWarp=utils.warp(img, biggest, wP,hP)
        cv2.imshow('A4', imgWarp)
        img2, conts2 = utils.getContours(imgWarp, draw=False,cThreshold=[50,50], minArea=2000, filter=4)
        if(len(conts2)!=0):
            for obj in conts2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                newPoints=utils.reorder(obj[2])
                nW=round((utils.findDistance(newPoints[0][0]//scale,newPoints[1][0]//scale)/10),1)
                nH=round((utils.findDistance(newPoints[0][0] // scale, newPoints[2][0] // scale) / 10), 1)
                cv2.arrowedLine(img2, (newPoints[0][0][0], newPoints[0][0][1]),
                                (newPoints[1][0][0], newPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (newPoints[0][0][0],newPoints[0][0][1]),(newPoints[2][0][0],newPoints[2][0][1]),
                                (255,0,255),3,8,0,0.05)
                x,y,w,h=obj[3]
                # cv2.putText()
                cv2.putText(img2,'{}cm'.format(nW),(x+30,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
                cv2.putText(img2, '{}cm'.format(nH), (x -70, y+h//2), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 255), 2)
        cv2.imshow('A4', img2)
    img=cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
