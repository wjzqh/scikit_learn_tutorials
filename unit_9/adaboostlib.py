from numpy import *
from matplotlib import pyplot as plt
def loadDataSet(fileName):
	recordlist=[]
	fp = open(fileName,"rb")
	content=fp.read()
	fp.close()
	rowlist=content.splitlines()
	recordlist=[map(eval,row.split("\t")) for row in rowlist if row.strip()]
	dataSet=mat(recordlist)[:,:-1]
	labels =mat(recordlist)[:,-1].T
	return dataSet,labels
	
def plotROC(predStrengths,classLabels):
	cur=(1.0,1.0)
	ySum=0.0
	numPosClas=sum(array(classLabels)==1.0)
	yStep=1/float(numPosClas);xStep=1/float(len(classLabels)-numPosClas)
	sortedIndicies=predStrengths.argsort()
	fig=plt.figure()
	fig.clf()
	ax=plt.subplot(111)
	for index in sortedIndicies.tolist()[0]:
		#print "index:",index
		if classLabels[index] == 1.0:
			delX=0;delY=yStep
		else:
			delX=xStep;delY=0
			ySum+=cur[1]
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
		cur=(cur[0]-delX,cur[1]-delY)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False positive rate');plt.ylabel('True positive rate')
	plt.title('ROC curve for AdaBoost horse colic detection system')
	ax.axis([0,1,0,1])
	plt.show()
	print "the Area Under the Curve is:",ySum*xStep
	
def splitDataSet(dataMat,Column,threshVal,operator):
	retArray=ones((shape(dataMat)[0],1))
	if operator=='lt':
		retArray[dataMat[:,Column]<=threshVal]=-1.0
	else:
		retArray[dataMat[:,Column]>threshVal]=-1.0
	return retArray
	
def decisionTree(dataSet,labellist,D):
	dataMat=mat(dataSet);labelMat=mat(labellist).T
	m,n=shape(dataMat)
	numSteps=10.0
	bestFeat={}
	bestClass=mat(zeros((m,1)))
	minError=inf
	for i in xrange(n):
		rangeMin=dataMat[:,i].min()
		rangeMax=dataMat[:,i].max()
		stepSize=(rangeMax-rangeMin)/numSteps
		for j in xrange(-1,int(numSteps)+1):
			threshVal=(rangeMin+float(j)*stepSize)
			for operator in ['lt','gt']:
				predictedVals=splitDataSet(dataMat,i,threshVal,operator)
				errSet=mat(ones((m,1)))
				errSet[predictedVals==labelMat]=0
				weightedError=D.T*errSet
				if weightedError<minError:
					minError=weightedError
					bestClass=predictedVals.copy()
					bestFeat['dim']=i
					bestFeat['thresh']=threshVal
					bestFeat['oper']=operator
	return bestFeat,minError,bestClass
	
def adaBoostTrain(dataSet,labellist,numIt=40):
	#print "numIt:",numIt
	weakClassSet=[]
	m=shape(dataSet)[0]
	#print "shape(dataSet):",shape(dataSet)
	D=mat(ones((m,1))/m)
	aggClassSet=mat(zeros((m,1)))
	for i in xrange(numIt):
		bestFeat,error,EstClass=decisionTree(dataSet,labellist,D)
		#print "bestFeat:",bestFeat
		#print "error:",error
		#print "EstClass:",EstClass
		alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
		bestFeat['alpha']=alpha
		weakClassSet.append(bestFeat)
		wtx=multiply(-1*alpha*mat(labellist).T,EstClass)
		D=multiply(D,exp(wtx))
		D=D/D.sum()
		aggClassSet+=alpha*EstClass
		totalErr=multiply(sign(aggClassSet)!=mat(labellist).T,ones((m,1)))
		errorRate=totalErr.sum()/m
		if errorRate==0.0:break
	#print "totalError:",sum(totalErr)
	return weakClassSet,aggClassSet
    
def adaClassify(dataToClass,classifierArr):
	dataMatrix=mat(dataToClass)
	m=shape(dataMatrix)[0]
	aggClassEst=mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst=splitDataSet(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['oper'])
		aggClassEst+=classifierArr[i]['alpha']*classEst
	return sign(aggClassEst)
	
