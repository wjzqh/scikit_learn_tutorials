#-*- coding:utf-8 -*-

from numpy import *
import cv2
'''
win_name='mypicture'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
img=cv2.imread('mypicture.jpg',1)
cv2.imshow(win_name,img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("paulwalker.mono.pgm",img)

from matplotlib import pyplot as plt
img=cv2.imread('paulwalker.mono.pgm',0)
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([]),plt.yticks([])
plt.show()

img=zeros((512,512,3))
cv2.line(img,(0,255),(512,255),(0,0,255),2)
cv2.rectangle(img,(150,150),(350,350),(255,255,0),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img=zeros((512,512,3))
cv2.circle(img,(255,350),60,(0,255,255),2)
cv2.ellipse(img,(255,150),(100,50),0,0,360,(255,255,0),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img=zeros((512,512,3))
pts=array([[50,100],[150,150],[170,120],[250,210],[250,310]])
cv2.polylines(img,[pts],True,(0,255,255),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
img=zeros((512,512,3))
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Face Cognition',(10,255),font,2,(255,255,255),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()