#-*- coding:UTF-8 -*-
from numpy import *
import numpy as np

class Kohonen(object):
	def __init__(self):
		self.lratemax=0.8;
		self.lratemin=0.05
		self.rmax=5.0;
		self.rmin=0.5;
		self.Steps=1000
		self.lratelist=[]
		self.rlist=[]
		self.w=[]
		self.M=2
		self.N=2
		self.dataMat=[]
		self.classLabel=[]

	def normalize(self,dataMat):
		[m,n]=shape(dataMat)
		for i in xrange(n-1):
			dataMat[:,i]=(dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
		return dataMat
		
	def distEclud(self,A,B):
		m,n=shape(B)
		Eclu=[]
		for i in xrange(n):
			dd = sqrt(sum(square(A-B[:,i].T)))
			Eclu.append(dd)
		return mat(Eclu)
		
	def loadDataSet(self,fileName):
		numFeat=len(open(fileName).readline().split('\t'))-1
		fr=open(fileName)
		for line in fr.readlines():
			lineArr=[]
			curLine=line.strip().split('\t')
			lineArr.append(float(curLine[0]));
			lineArr.append(float(curLine[1]));
			self.dataMat.append(lineArr)
		self.dataMat=mat(self.dataMat)

	def init_grid(self):
		k=0
		grid=mat(zeros((self.M*self.N,2)))
		for i in xrange(self.M):
			for j in xrange(self.N):
				grid[k,:]=[i,j]
				k += 1;
		#print grid
		return grid
		
	def ratecalc(self,i):
		Learn_rate = self.lratemax - (i+1.0)*(self.lratemax-self.lratemin)/self.Steps
		R_rate = self.rmax - (i+1.0)*(self.rmax-self.rmin)/self.Steps
		return Learn_rate,R_rate
	
	def train(self):
		dm,dn=shape(self.dataMat)
		normDataset=self.normalize(self.dataMat)
		grid=self.init_grid()
		self.w=random.rand(dn,self.M*self.N)
		distM=self.distEclud
		#迭代求解
		if self.Steps<5*dm:self.Steps=5*dm
		for i in xrange(self.Steps):
			lrate,r=self.ratecalc(i)
			#print lrate,r
			self.lratelist.append(lrate);self.rlist.append(r)
			k=random.randint(0,dm)
			mySample=normDataset[k,:]
			#print mySample,mySample[0]
			minIndx=(distM(mySample,self.w)).argmin()
			#print mySample,self.w,distM(mySample,self.w),minIndx
			d1=ceil(minIndx/self.M)
			d2=mod(minIndx,self.M)
			#print "minIndx:%d,d1:%d,d2:%d"%(minIndx,d1,d2)
			distMat=distM(mat([d1,d2]),grid.T)
			nodelindx=(distMat<r).nonzero()[1]
			#print (distMat<r).nonzero(),nodelindx
			for j in xrange(shape(self.w)[1]):
				if sum(nodelindx==j):
					self.w[:,j]=self.w[:,j]+lrate*(mySample[0]-self.w[:,j])
		self.classLabel=range(dm)
		for i in xrange(dm):
			self.classLabel[i]=distM(normDataset[i,:],self.w).argmin()
		self.classLabel=mat(self.classLabel)

	def showCluster(self,plt):
		lst=unique(self.classLabel.tolist()[0])
		#print self.classLabel.tolist()[0],lst
		i=0
		for cindx in lst:
			myclass=nonzero(self.classLabel==cindx)[1]
			#print nonzero(self.classLabel==cindx),myclass
			xx=self.dataMat[myclass].copy()
			if i ==0:plt.plot(xx[:,0],xx[:,1],'bo')
			elif i ==1:plt.plot(xx[:,0],xx[:,1],'rd')
			elif i ==2:plt.plot(xx[:,0],xx[:,1],'gD')
			elif i ==3:plt.plot(xx[:,0],xx[:,1],'c^')
			i+=1
		plt.show()