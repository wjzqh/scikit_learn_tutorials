from numpy import *
import copy
class BoltzmannNet(object):
	def __init__(self):
		self.dataMat=[]
		self.MAX_ITER=2000
		self.T0=2000
		self.Lambda=0.97
		self.iteration=0
		self.dist=[]
		self.pathindx=[]
		self.bestdist=0
		self.bestpath=[]
		
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

	def distEclud(self,A,B):
		m,n=shape(B)
		x,y=shape(A)
		Eclu=[]
		for j in xrange(x):
			for i in xrange(n):
				dd = sqrt(sum(square(A[j,:]-B[:,i].T)))
				Eclu.append(dd)
		#print Eclu
		return mat(Eclu).reshape(x,n)
		
	def boltzmann(self,newl,oldl,T):
		return exp(-(newl-oldl)/T)
		
	def pathLen(self,dist,path):
		N=len(path)
		plen=0
		#print N,dist,path
		for i in xrange(0,N-1):
			plen+=dist[path[i],path[i+1]]
		plen+=dist[path[0],path[N-1]]
		return plen
		
	def changePath(self,old_path):
		N=len(old_path)
		if random.rand()<0.25:
			chpos=floor(random.rand(1,2)*N).tolist()[0]
			new_path=copy.deepcopy(old_path)
			new_path[int(chpos[0])]=old_path[int(chpos[1])]
			new_path[int(chpos[1])]=old_path[int(chpos[0])]
		else:
			d=ceil(random.rand(1,3)*N).tolist()[0];d.sort()
			a=int(d[0]);b=int(d[1]);c=int(d[2])
			if a!=b and b!=c:
				new_path=copy.deepcopy(old_path)
				new_path[a:c-1]=old_path[b-1:c-1]+old_path[a:b-1]
			else:
				new_path=self.changePath(old_path)
		return new_path
		
	def drawPath(self,Seq,dataMat,plt,color='b'):
		m,n=shape(dataMat)
		#print Seq
		px=(dataMat[Seq,0]).tolist()
		py=(dataMat[Seq,1]).tolist()
		#print px,py
		px.append(px[0]);py.append(py[0])
		plt.plot(px,py,color)
		
	def drawScatter(self,plt):
		px=(self.dataMat[:,0]).tolist()
		py=(self.dataMat[:,1]).tolist()
		plt.scatter(px,py,c='green',marker='o',s=60)
		i=65
		for x,y in zip(px,py):
			plt.annotate(str(chr(i)),xy=(x[0]+40,y[0]),color='black')
			i+=1
			
	def TrendLine(self,plt,color='b'):
		plt.plot(range(len(self.dist)),self.dist,color)
	
	def initBMNet(self,m,n,distMat):
		self.pathindx=range(m)
		random.shuffle(self.pathindx)
		self.dist.append(self.pathLen(distMat,self.pathindx))
		return self.T0,self.pathindx,m
			
	def train(self):
		[m,n]=shape(self.dataMat)
		print self.dataMat
		distMat=self.distEclud(self.dataMat,self.dataMat.T)
		[T,curpath,MAX_M]=self.initBMNet(m,n,distMat)
		step=0
		while step<=self.MAX_ITER:
			m=0
			while m<=MAX_M:
				curdist=self.pathLen(distMat,curpath)
				newpath=self.changePath(curpath)
				newdist=self.pathLen(distMat,newpath)
				if(curdist>newdist):
					curpath=newpath
					self.pathindx.append(curpath)
					self.dist.append(newdist)
					self.iteration+=1
				else:
					if random.rand()<self.boltzmann(newdist,curdist,T):
						curpath=newpath
						self.pathindx.append(curpath)
						self.dist.append(newdist)
						self.iteration+=1
				m+=1
			step+=1
			T=T*self.Lambda
		self.bestdist=min(self.dist)
		indxes=argmin(self.dist)
		self.bestpath=self.pathindx[indxes]
		
		