#-*- coding:utf-8 -*-
from numpy import *
import numpy as np
import sys,os
import copy
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt

class Eigenfaces(object):
	def __init__(self):
		self.eps=1.0e-16
		self.X=[]
		self.y=[]
		self.Mat=[]
		self.eig_v=0
		self.eig_vect=0
		self.mu=0
		self.projections=[]
		self.dist_metric=0
		
	def compute(self):
		self.PCA(300)
		for xi in self.X:
			self.projections.append(self.project(xi.reshape(1,-1)))
			
	def distEclud(self,vecA,vecB):
		return linalg.norm(vecA-vecB)+self.eps
		
	def cosSim(self,vecA,vecB):
		return (dot(vecA,vecB.T)/((linalg.norm(vecA)*linalg.norm(vecB))+self.eps))[0,0]
		
	def project(self,XI):
		if self.mu is None: return np.dot(XI,self.eig_vect)
		return np.dot(XI-self.mu,self.eig_vect)
		
	def subplot(slef,title,images):
		fig=plt.figure()
		fig.text(.5,.95,title,horizontalalignment='center')
		for i in xrange(len(images)):
			ax0=fig.add_subplot(4,4,(i+1))
			plt.imshow(asarray(images[i]),cmap="gray")
			plt.xticks([]),plt.yticks([])
		plt.show()
	
	def loadimgs(self,path):
		classlabel=0
		for dirname,dirnames,filenames in os.walk(path):
			#print "#######1111111,dirname:%s,dirnames:%s"%(dirname,dirnames)
			for subdirname in dirnames:
				#print "#######2222222,subdirname:",subdirname
				sub_path=os.path.join(dirname,subdirname)
				for filename in os.listdir(sub_path):
					#print "#######33333333,filename:",filename
					im=Image.open(os.path.join(sub_path,filename))
					im=im.convert("L")
					self.X.append(np.asarray(im,dtype=np.uint8))
					self.y.append(classlabel)
				classlabel+=1
		
	def genRowMatrix(self):
		#print "self.X:",self.X
		#print "size:",self.X[1].size
		self.Mat=np.empty((0,self.X[0].size),dtype=self.X[0].dtype)
		#print "shape(self.Mat):",shape(self.Mat)
		for row in self.X:
			#print "row:",row
			self.Mat=np.vstack((self.Mat,np.asarray(row).reshape(1,-1)))
		#print "shape(self.Mat):",shape(self.Mat)
		#print "self.Mat:",self.Mat
			
	def PCA(self,k=0):
		self.genRowMatrix()
		[n,d]=shape(self.Mat)
		print "n:%d,d:%d"%(n,d)
		if(k>n): k=n
		self.mu=self.Mat.mean(axis=0)
		print shape(self.Mat),shape(self.mu)
		self.Mat=self.Mat-self.mu
		if n>d:
			XTX=np.dot(self.Mat.T,self.Mat)
			[self.eig_v,self.eig_vect]=linalg.eigh(XTX)
		else:
			XTX=np.dot(self.Mat,self.Mat.T)
			[self.eig_v,self.eig_vect]=linalg.eigh(XTX)
		#print "self.eig_v:",self.eig_v
		self.eig_vect=np.dot(self.Mat.T,self.eig_vect)
		for i in xrange(n):
			self.eig_vect[:,i]=self.eig_vect[:,i]/linalg.norm(self.eig_vect[:,i])
		idx=np.argsort(-self.eig_v)
		#print "idx:",idx
		self.eig_v=self.eig_v[idx]
		self.eig_vect=self.eig_vect[:,idx]
		self.eig_v=self.eig_v[0:k].copy()
		self.eig_vect=self.eig_vect[:,0:k].copy()
		print "shape(self.eig_v):",shape(self.eig_v)
		print "shape(self.eig_vect):",shape(self.eig_vect)
		
	def predict(self,XI):
		minDist = np.finfo('float').max
		minClass = -1
		Q = self.project(XI.reshape(1,-1))
		#print "Q:",Q
		for i in xrange(len(self.projections)):
			dist = self.dist_metric(self.projections[i],Q)
			if dist < minDist:
				minDist = dist
				minClass = self.y[i]
		return minClass