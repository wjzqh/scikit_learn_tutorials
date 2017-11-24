import numpy as np
def loadDataSet():
	postingList=[['my','dog','has','flea','problems','help','please'],
							['maybe','not','take','him','to','dog','park','stupid'],
							['my','dalmation','is','so','cute','I','love','him','my'],
							['stop','posting','stupid','worthless','garbage'],
							['mr','licks','ate','my','steak','how','to','stop','him'],
							['quit','buying','worthless','dog','food','stupid']]
							
	classVec = [0,1,0,1,0,1]  #1 is abusive,0 not
	return postingList,classVec
	
class NBayes(object):
	def __init__(self):
			self.vocabulary = []
			self.idf = 0
			self.tf = 0
			self.tdm=0
			self.Pcates={}
			self.labels=[]
			self.doclength=0
			self.vocablen=0
			self.testset=0
			
	def train_set(self,trainset,classVec):
			self.cate_prob(classVec)
			self.doclength=len(trainset)
			tempset=set()
			[tempset.add(word) for doc in trainset for word in doc]
			self.vocabulary=list(tempset)
			#print "self.vocabulary:",self.vocabulary
			self.vocablen=len(self.vocabulary)
			self.calc_wordfreq(trainset)
			#print "self.tf:",self.tf
			#print "self.idf:",self.idf
			self.build_tdm()
			#print "self.tdm:",self.tdm
			
	def cate_prob(self,classVec):
			self.labels=classVec
			labeltemps=set(self.labels)
			for labeltemp in labeltemps:
				self.Pcates[labeltemp]=float(self.labels.count(labeltemp))/float(len(self.labels))
				
	def calc_wordfreq(self,trainset):
			self.idf=np.zeros([1,self.vocablen])
			self.tf=np.zeros([self.doclength,self.vocablen])
			for indx in xrange(self.doclength):
				for word in trainset[indx]:
					self.tf[indx,self.vocabulary.index(word)]+=1
				for signleword in set(trainset[indx]):
					self.idf[0,self.vocabulary.index(signleword)]+=1
					
	def build_tdm(self):
			self.tdm=np.zeros([len(self.Pcates),self.vocablen])
			sumlist=np.zeros([len(self.Pcates),1])
			for indx in xrange(self.doclength):
					self.tdm[self.labels[indx]]+=self.tf[indx]
					sumlist[self.labels[indx]]=np.sum(self.tdm[self.labels[indx]])
			self.tdm=self.tdm/sumlist
			
	def map2vocab(self,testdata):
			self.testset=np.zeros([1,self.vocablen])
			for word in testdata:
				self.testset[0,self.vocabulary.index(word)]+=1
				
	def predict(self,testset):
			if np.shape(testset)[1]!=self.vocablen:
				print "输入错误"
				exit(0)
			predvalue=0
			predclass=""
			#print "zip(self.tdm,self.Pcates):",zip(self.tdm,self.Pcates)
			for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
				#print "#####tdm_vect:%s,keyclass:%d"%(tdm_vect,keyclass)
				#print "testset*tdm_vect:",testset*tdm_vect
				#print "self.Pcates[keyclass]:",self.Pcates[keyclass]
				#print "testset*tdm_vect*self.Pcates[keyclass]:",testset*tdm_vect*self.Pcates[keyclass]
				temp = np.sum(testset*tdm_vect*self.Pcates[keyclass])
				if temp>predvalue:
					predvalue=temp
					predclass=keyclass
			return predclass
