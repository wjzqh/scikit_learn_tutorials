#-*- coding:utf-8 -*-
from numpy import *
import cv2

face_cascade=cv2.CascadeClassifier('E:\\scikit_learn\\unit_9\\lbpcascade_frontalface.xml')
img=cv2.imread('IMG_20170923_165156.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.2,3)
for (x,y,w,h) in faces:
	img2=cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
	roi_gray=gray[y:y+h,x:x+w]
	roi_color=img[y:y+h,x:x+w]
	
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("paulwalker.head.jpg",img)