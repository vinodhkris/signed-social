#This File is to get the beta which is the parameter for P(Y). After this, we need to estimate the J and then write the maximisation

from __future__ import division
from collections import defaultdict
import util
import numpy as np 
import scipy 
import networkGraphConstruction as ngc
import cPickle as pickle
import constructNoiseGraph as cng
from scipy import stats
import math
import scipy.optimize as scopt
import numpy as np 
import random
from scipy.misc import logsumexp
import sys

print 'Enter mode you want to calculate accuracy of: '
print '1. Only Linguistic features(Theta)\n2. Linguistic and structure features (Theta & Beta)\n3. Linguistic, structure and common friends feature (Theta, Beta, Alpha)\n4. Linguistic and common friends feature (Theta, Alpha)\n5. Only structure'
value = raw_input('Enter choice: ')
mode = int(value)

nc = 2 #number of clusters 0 - cluster 1, 1 - cluster 2
betaR = float(sys.argv[1])
alphaR = float(sys.argv[2])
thetaR = float(sys.argv[3])
threshold = int(sys.argv[4])

def unpickle(filename):
	f = open(filename,"rb") 
	heroes = pickle.load(f)
	return heroes

def tupleToIndex(tup):
	if tup[0] == 0 and tup[1] == 0 and tup[2] == 0:
		return 0
	elif tup[0] == 1 and tup[1] == 1 and tup[2] == 1:
		return 3
	elif (tup[0] == 1 and tup[1] == 1) or (tup[1] == 1 and tup[2] == 1) or (tup[2] == 1 and tup[0] == 1):
		return 2
	else:
		return 1
	#return (nc**0)*tup[0] + (nc**1)*tup[1] + (nc**2)*tup[2]

#yij - relationship value
#qyij,qyjk, qyki - expected relationship value yij yjk yki - relationship values - numofedges for each q
#beta - 27 value array
#Theta - 3*number of address terms - 3*123 matrix
#Dn basically address term to index mapping

#To get the output vector which basically is the address terms said between 2 nodes - check outputVector.txt
def getVector(filename):
	file1 = open(filename,"r")
	addressTermsVector = defaultdict()	#addressTermsVector - dictionary with key as the edge tuple and which in turn is a dictionary with key as the word and value as the number of times that word has been said between the 2 nodes
	for line in file1:
		row = line.split(";")
		try:
			addressTermsVector[(row[0].split(",")[0],row[0].split(",")[1])][row[1]] = int(row[2].rstrip('\'\"-,.:;!?\n '))
		except:
			addressTermsVector[(row[0].split(",")[0],row[0].split(",")[1])] = defaultdict()
			addressTermsVector[(row[0].split(",")[0],row[0].split(",")[1])][row[1]] = int(row[2].rstrip('\'\"-,.:;!?\n '))
	return addressTermsVector


###########################################Noise Contrastive Estimation functions###############################

#For degree feature - alphadegree - values for noise contrastive estimation used in the maximisation step to approximate values for the feature parameters (we assumed a exponential distribution)
def lnpmxalphadegree(alphadegree,c,edges,qyij,characterDegree,maxDegree):
	sum2 = 0.0
	for edge in edges:	
		edgeq = edge
		if edge not in qyij:
			edgeq = (edge[1],edge[0])
		for m in xrange(nc):
			#if characterDegree[edge[0]] == 1 or characterDegree[edge[1]] == 1:
			if edge[0] in maxDegree or edge[1] in maxDegree:
				sum2+=qyij[edgeq][m]*alphadegree[m]
	pm = sum2
	return pm+c

def lnpmyalphadegree(alphadegree,c,edges,polarity,characterDegree,maxDegree):
	sum2 = 0.0
	for edge in edges:
		edgep = edge
		if edge not in polarity:
			edgep = (edgep[1],edgep[0])
		#if characterDegree[edge[0]] == 1 or characterDegree[edge[1]] == 1:
		if edge[0] in maxDegree or edge[1] in maxDegree:
			sum2+=alphadegree[polarity[edgep]]
	
	pm = sum2
	return pm+c

def lllnpmxalphadegree(alphadegree,c,edges,qyij,characterDegree,maxDegree):
	sum2 = 0.0
	for i in xrange(len(edges)):
		for edge in edges[i]:	
			edgeq = edge
			if edge not in qyij:
				edgeq = (edge[1],edge[0])
			for m in xrange(nc):
				#if characterDegree[edge[0]] == 1 or characterDegree[edge[1]] == 1:
				if edge[0] in maxDegree or edge[1] in maxDegree:
					sum2+=qyij[edgeq][m]*alphadegree[m]
	pm = sum2
	return pm+c


#For alpha - common friends feature
def lnpmxalpha(alpha,c,edges,qyij,commonFriends):
	sum2 = 0.0
	for edge in edges:	
		edgeCommonFriends = edge
		edgeq = edge
		if edge not in commonFriends:
			edgeCommonFriends = (edge[1],edge[0])
		if edge not in qyij:
			edgeq = (edge[1],edge[0])
		for m in xrange(nc):
			sum2+=commonFriends[edgeCommonFriends]*qyij[edgeq][m]*alpha[m]
	pm = sum2
	return pm+c


def lnpmyalpha(alpha,c,edges,polarity,commonFriends):
	sum2 = 0.0
	for edge in edges:
		edgeCommonFriends = edge
		edgep = edge
		if edge not in commonFriends:
			edgeCommonFriends = (edge[1],edge[0])
		if edge not in polarity:
			edgep = (edgep[1],edgep[0])
		sum2+=commonFriends[edgeCommonFriends]*alpha[polarity[edgep]]
	
	pm = sum2
	return pm+c

#pmx for the noise contrastive estimation
def lnpmx(beta,c,triads,qyij):
	sum1 = 0
	sum2 = 0
	#Partition function implementation
	#Requirements  - genearate noise and then generate the categorical distribution of the data for it. 
	#replace it entirely by the lbfgs implementation we derived
	for word in triads:
		for i in xrange(nc):
			for j in xrange(nc):
				for k in xrange(nc):
					if (word[0],word[1]) in qyij:
						q1 = qyij[(word[0],word[1])]
					else:
						q1 = qyij[(word[1],word[0])]

					if (word[1],word[2]) in qyij:
						q2 = qyij[(word[1],word[2])]
					else:
						q2 = qyij[(word[2],word[1])]

					if (word[2],word[0]) in qyij:
						q3 = qyij[(word[2],word[0])]
					else:
						q3 = qyij[(word[0],word[2])]
					
				#	print i,j,k,tupleToIndex((i,j,k))
					sum1+=beta[tupleToIndex((i,j,k))]*q1[i]*q2[j]*q3[k]
	pm = sum1
	return pm+c

#For estimating the pm value for the noise
def lnpmy(beta,c,triads,polarity):
	sum1 = 0
	sum2 = 0
	for word in triads:
		if (word[0],word[1]) in polarity:
			pol1 = polarity[(word[0],word[1])]
		else:
			pol1 = polarity[(word[1],word[0])]

		if (word[1],word[2]) in polarity:
			pol2 = polarity[(word[1],word[2])]
		else:
			pol2 = polarity[(word[2],word[1])]

		if (word[2],word[0]) in polarity:
			pol3 = polarity[(word[2],word[0])]
		else:
			pol3 = polarity[(word[0],word[2])]

		sum1+=beta[tupleToIndex((pol1,pol2,pol3))]
	
	pm = sum1
	return pm+c


def lnpn(polarity):
	pk = [0.50,0.50]			#Only 2 clusters
	numofedgestype0 = 0
	numofedgestype1 = 0
	for word in polarity:
		if polarity[word] == 0:
			numofedgestype0+=1
		if polarity[word] == 1:
			numofedgestype1+=1
	return numofedgestype0*pk[0] + numofedgestype1*pk[1] 


def sigmoid(t):
	return 1/(1+np.exp(-t))

def g(lnpm,lnpn):
	return lnpm - lnpn

def h(lnpm,lnpn):
	return sigmoid(g(lnpm,lnpn))

#The objective function given by noise contrastive estimation to get beta
def objectiveFunction(params, *args):
	sum1 = 0
	triads = args[0]
	T = len(triads)
	q = args[1]
	polarity = args[2]
	beta = params[:-1]
	#print beta
	c = params[-1]
	for t in xrange(T):
		lnpmxx = lnpmx(beta,c,triads[t],q)
		lnpmyy = lnpmy(beta,c,triads[t],polarity[t])
		lnpnx = lnpn(polarity[t])
		#print h(lnpmxx,lnpnx),1-h(lnpmyy,lnpnx)
		sum1 += math.log(h(lnpmxx,lnpnx)) + math.log(1-h(lnpmyy,lnpnx))
	sum1-=betaR*sum(beta**2)
	print -1*sum1/2*T
	return -1*sum1/2*T

#The objective function given by noise contrastive estimation to get alpha for common friends feature
def objectiveFunctionAlpha(params, *args):
	sum1 = 0
	edges = args[0]
	E = len(edges)
	q = args[1]
	polarity = args[2]
	commonFriends = args[3]
	alpha = params[:-1]
	c = params[-1]
	for t in xrange(E):
		lnpmxx = lnpmxalpha(alpha,c,edges[t],q,commonFriends)
		lnpmyy = lnpmyalpha(alpha,c,edges[t],polarity[t],commonFriends)
		lnpnx = lnpn(polarity[t])
		sum1 += math.log(h(lnpmxx,lnpnx)) + math.log(1-h(lnpmyy,lnpnx))

	sum1-=alphaR*sum(alpha**2)
	print -1*sum1/2*E
	return -1*sum1/2*E

#The objective function given by noise contrastive estimation to get alpha for characters degree feature
def objectiveFunctionAlphaDegree(params, *args):
	sum1 = 0
	edges = args[0]
	E = len(edges)
	q = args[1]
	polarity = args[2]
	characterDegree = args[3]
	maxDegree = args[4]
	alpha = params[:-1]
	c = params[-1]
	for t in xrange(E):
		lnpmxx = lnpmxalphadegree(alpha,c,edges[t],q,characterDegree,maxDegree)
		lnpmyy = lnpmyalphadegree(alpha,c,edges[t],polarity[t],characterDegree,maxDegree)
		lnpnx = lnpn(polarity[t])
	#	print 'lnpmxx',lnpmxx,'lnpmyy',lnpmyy
		sum1 += math.log(h(lnpmxx,lnpnx)) + math.log(1-h(lnpmyy,lnpnx))

	sum1-=sum(alpha**2)
	print sum1/2*E
	return -1*sum1/2*E

##########################################################Ended Noise Contrastive functions##########################

######################################Functions for evaluating the log likelihood###########################

#For common friends, to compute the log likelihood of the data given the common friends parameter. 
#The log likelihood of the data becomes the sum of the log of numerator of the probability function 
#+ a parameter for the partition function
def lllnpmxalpha(alpha,c,edges,qyij,commonFriends):
	sum2 = 0.0
	for i in xrange(len(edges)):
		for edge in edges[i]:	
			edgeCommonFriends = edge
			edgeq = edge
			if edge not in commonFriends:
				edgeCommonFriends = (edge[1],edge[0])
			if edge not in qyij:
				edgeq = (edge[1],edge[0])
			for m in xrange(nc):
				sum2+=commonFriends[edgeCommonFriends]*qyij[edgeq][m]*alpha[m]
	pm = sum2
	return pm+c

#For the original beta - for estimating the log likelihood function
#The log likelihood of the data becomes the sum of the log of numerator of the probability function 
#+ a parameter for the partition function
def lllnpmx(beta,c,triads,qyij):
	sum1 = 0
	sum2 = 0
	print len(triads)
	#Partition function implementation
	#Requirements  - genearate noise and then generate the categorical distribution of the data for it. 
	#replace it entirely by the lbfgs implementation we derived
	for m in xrange(len(triads)):
		for word in triads[m]:
			for i in xrange(nc):
				for j in xrange(nc):
					for k in xrange(nc):
						if (word[0],word[1]) in qyij:
							q1 = qyij[(word[0],word[1])]
						else:
							q1 = qyij[(word[1],word[0])]

						if (word[1],word[2]) in qyij:
							q2 = qyij[(word[1],word[2])]
						else:
							q2 = qyij[(word[2],word[1])]

						if (word[2],word[0]) in qyij:
							q3 = qyij[(word[2],word[0])]
						else:
							q3 = qyij[(word[0],word[2])]
						
						sum1+=beta[tupleToIndex((i,j,k))]*q1[i]*q2[j]*q3[k]
		
	pm = sum1
	return pm+c

#This is for the equation : 
#summation over all edges (summation over all types of edge (q * sum over all address terms (log(thetasum[that type of edge])))
def evaluatelogpxy(qyij,theta,nc,outputVector):
	logpxy = 0.0
	for edge in qyij:
		for clusternum in xrange(nc):
			thetasum = 0
			for addressterm in outputVector[edge]:
				if outputVector[edge][addressterm]!=0:		#if they spoke that word
					thetasum+=math.log(theta[addressterm][clusternum])*outputVector[edge][addressterm]
			logpxy+=qyij[edge][clusternum]*thetasum  #sum over all edges (sum over all types of edge (q)*thetasum for that type of edge)
	return logpxy	

#sum(qlogq)
def qlogq(qyij,nc):
	sum1 = 0
	for edge in qyij:
		for m in range(nc):
			sum1+=qyij[edge][m]*math.log(qyij[edge][m])
	return sum1

#sum(sum(q)-1)
def normalizell(qyij,nc):
	sum1 = 0
	sum2 = 0
	for edge in qyij:
		sum1 = 0
		for m in range(nc):
			sum1+=qyij[edge][m]
		sum2+=(sum1 - 1)
	return sum2

############################################Ending Log likelihood functions ################################

#Initialisation of parameters

#Feature files
commonFriendsDict = unpickle("commonFriends.txt")
characterDegreeDict = unpickle("characterDegree.txt")
maxDegreeDict = unpickle("maxDegree.txt")

while True:
	try:
		#Declaration of all the parameter variables
		beta = np.zeros(4)
		for i in xrange(4):
			beta[i] = random.random() 						#random initialisation of beta

		alpha = np.zeros(nc)

		alphadegree = np.zeros(nc)
		alphasum = 0
		alphadegreesum = 0

		for i in xrange(nc-1):
			alpha[i] = random.random()
			alphadegree[i] = random.random()
			alphasum+=alpha[i]
			alphadegreesum+=alphadegree[i]

		alpha[nc-1] = 1 - alphasum
		alphadegree[nc-1] = 1 - alphadegreesum

		placeHolderTerms = []
		addressTerms = []
		for word in open("placeHolders.txt","r"):
			placeHolderTerms.append(word.strip())
		for word in open("addressTerms.txt","r"):
			addressTerms.append(word.strip())
		print len(placeHolderTerms),len(addressTerms)
		possibleTitles = {"lastName","firstName","firstName+lastName"}
		for word in addressTerms:
			word = word.rstrip('\'\"-,.:;!?\n ')
			possibleTitles.add(word)
			possibleTitles.add(word+"+firstName")
			possibleTitles.add(word+"+lastName")
			possibleTitles.add(word+"+firstName+lastName")

		for word in placeHolderTerms:
			word = word.rstrip('\'\"-,.:;!?\n ')
			possibleTitles.add(word)
		print len(possibleTitles),possibleTitles
		g1 = ngc.Graph()
		triads = g1.getTriads()	#dictionary of triads
		nodes = g1.getNodes()
		edges = g1.getEdges()	#dictionary of edges
		outputVector = getVector("outputVector_improved.txt") 				#to get output vector - number of address termsX number of edges
		opv = getVector("outputVector.txt")
		loglikelihoods = []

		print len(outputVector),len(opv)
		#Collapsing titles
		addressTermCounts = defaultdict(int)
		for edge in opv:
			if edge not in outputVector:
				outputVector[edge] = defaultdict(int)
		print len(outputVector)
		for edge in outputVector:
			for addressterm in outputVector[edge]:
				addressTermCounts[addressterm]+=outputVector[edge][addressterm]


		filteredAddresses = []
		filteredPlaceHolders = []
		for addressterm in addressTermCounts:
			if addressTermCounts[addressterm] <threshold:
				if addressterm in placeHolderTerms:
					filteredPlaceHolders.append(addressterm)
				else:
					filteredAddresses.append(addressterm)					#getting all the address terms that should be filtered

		print 'Collapsed addressterms are ',filteredAddresses,filteredPlaceHolders

		for edge in outputVector:
			outputVector[edge]['collapsedTitle'] = 0
			outputVector[edge]['collapsedPlaceHolder'] = 0
			for addressterm in filteredAddresses:
				if addressterm in outputVector[edge]:
					if 'collapsedTitle' not in outputVector[edge]:
						outputVector[edge]['collapsedTitle']=outputVector[edge][addressterm]
					else:	
						outputVector[edge]['collapsedTitle']+=outputVector[edge][addressterm]  			#Collapsed title in outputVector
					outputVector[edge][addressterm] = 0
			for addressterm in filteredPlaceHolders:
				if addressterm in outputVector[edge]:
					if 'collapsedPlaceHolder' not in outputVector[edge]:
						outputVector[edge]['collapsedPlaceHolder']=outputVector[edge][addressterm]
					else:	
						outputVector[edge]['collapsedPlaceHolder']+=outputVector[edge][addressterm]  			#Collapsed title in outputVector
					outputVector[edge][addressterm] = 0
					
		copyPossibleTitles = set(possibleTitles)
		for addressterm in copyPossibleTitles:
			if addressterm in filteredAddresses or addressterm in filteredPlaceHolders:
				possibleTitles.remove(addressterm) 					#Removed addressterms in possibletitles

		possibleTitles.add('collapsedTitle')
		possibleTitles.add('collapsedPlaceHolder')

		print len(filteredAddresses),len(filteredPlaceHolders),len(possibleTitles)

		filteredAddressesPickle = open("filteredAddressesPickle.txt","wb")
		pickle.dump(filteredAddresses,filteredAddressesPickle)
		filteredAddressesPickle.close()

		filteredPlaceHolderPickle = open("filteredPlaceHoldersPickle.txt","wb")
		pickle.dump(filteredPlaceHolders,filteredPlaceHolderPickle)
		filteredPlaceHolderPickle.close()

		#theta is - for each word theta[word][0] value for that word for </> and theta[word][1] for that word is =
		theta = defaultdict()
		possibleTitlesLengthDenominator = len(possibleTitles)*0.50	#Assuming that the initial value is 0.5 
		thetaword1sum = 0.0
		thetaword2sum = 0.0
		for word in possibleTitles:
			a = random.random() 					#randomise both of them, 
			b = random.random()
			theta[word] = [a,b]#theta[word] = [0.4,0.6]
			thetaword1sum+=a
			thetaword2sum+=b
		for word in theta:
			theta[word][0] = theta[word][0]/thetaword1sum
			theta[word][1] = theta[word][1]/thetaword2sum

		#Initialise q
		qyij = defaultdict()
		print "length of output vector",len(outputVector)
		print "Edges",len(edges),len(nodes),len(triads)

		#	raw_input()
		#raw_input()
		#Expectation step
		#qyij = k*exp(sum(k:i,j,k in triads) qyjk qyki betaijk + sum (for all words) log(ThetayijDn) )
		print "Edges are : " 
		for edges in outputVector:
			qyij[edges] = [0.5]*nc

		print "Initialized qyij ",len(qyij)
		#qyij - dictionary of edges for each type of relationship
		#expectation part
		#calculate q for every pair of edges
		c1 = 1.0	#initial c
		alphac = 1.0
		alphadegreec = 1.0
		print 'Starting'
		#raw_input('Press any key to continue')
		for number in xrange(20):	#number of iterations for the expectation maximisation

			if mode!=5:
				print theta
				print "Theta"
		#	raw_input('Press any key to continue')
			if mode!=1:
				if mode!=4:
					print beta,c1
					print "Beta,c"
			#	raw_input('Press any key to continue')
				if mode!=2:
					print alpha,alphac
					print "Alpha,c"

			print "iteration",number
		#	raw_input('Press any key to continue')
			print "THe expectation function"

			print 'qyij updating'
			for edges in qyij:			#Iterate through the edges of the graph
				#sum(k:i,j,k in triads) qyjk qyki betaijk
				denominator = 0.0
				sum1 = [0]*nc			#for a pair of nodes, for each type of edge, and over all triads
				sum2 = [0]*nc
				numerator = 0

				if mode != 1: 											#Not only linguistic features
					if mode!=4:
						for triad in triads:
							if edges[0] in triad and edges[1] in triad:	#find the triad where these nodes are
							#	print "Node1 : ",word1[0],"Node 2 : ",word1[1],"Triad : ",word
								k = tuple(set(triad) - set((edges[0],edges[1])))[0]		#The third edge in the triad
								firstvar = 0.0
								secondvar = 0.0
								if (k,edges[0]) not in qyij:		#anyway it is undirected edges so direction does not matter
									firstvar = qyij[(edges[0],k)]	#qyki
								else:
									firstvar = qyij[(k,edges[0])]
								if (k,edges[1]) not in qyij:
									secondvar = qyij[(edges[1],k)]		#qyjk
								else:
									secondvar = qyij[(k,edges[1])]
							#	print "firstvar ",firstvar," Second Var : ",secondvar
								
								for m in xrange(nc):									#for each type of yij for a pair of nodes
									for i in xrange(nc):								#Sum over all yjk and yki
										for j in xrange(nc):
											sum1[m]+=firstvar[i]*secondvar[j]*beta[tupleToIndex((m,i,j))]
							
					if mode!=2:
						for m in xrange(nc):		
							sum1[m]+=commonFriendsDict[edges]*alpha[m]

				if mode!=5: 										#Only structure features	
					for addressterm in outputVector[edges]:					#does not depend on the type of edge - for all address terms that is in the conversation between them
						if outputVector[edges][addressterm]!=0:			#if they had spoken that address term as part of the conversation
							for m in xrange(nc):
								sum2[m]+=outputVector[edges][addressterm]*math.log(theta[addressterm][m])


				summ = [0]*nc
				for m in xrange(nc):
					summ[m] = sum1[m]+sum2[m]
				l = logsumexp(summ)
				for o in xrange(nc):
					qyij[edges][o] = np.exp(sum1[o]+sum2[o]-l)

			print 'qyij evaluated and updated'	

			#maximisation step
			#Theta derivation in terms of qyij
			#Reminder - Theta is words X 3
			#output vector - edgesX words
			#qyij - edges X 3
			#Therefore theta is sum total of all qyijs of that edge that spoke that word
			#outputvector : {edge,word:1}
			if mode!=5: 											#not only structure features
				print 'evaluating theta'
				for edges in outputVector:									#all edges
					for addressterm in outputVector[edges]:			#all words in that edge
						if outputVector[edges][addressterm]!=0:	#if they spoke that word
							for m in xrange(nc):	#SOFT NOT HARD 
								theta[addressterm][m]+=qyij[edges][m]*outputVector[edges][addressterm]		#for each type of word add that edge that spoke the word
							for m in xrange(nc):
								theta[addressterm][m]+=thetaR
				#normalizing factor
				denominator = [0.0]*nc
				for m in xrange(nc):
					for addressterm in theta:
						denominator[m]+=theta[addressterm][m]
					for addressterm in theta:
						theta[addressterm][m] = theta[addressterm][m]/denominator[m]

					
				print 'theta evaluated and updated'

			#Beta derivation in terms of qyij
			print 'Extracting noise data generated by graphsForEachMovie.py'
			edges = unpickle("edgesPickle.txt")
			polarity = unpickle("polarityPickle.txt")
			triad = unpickle("triadsPickle.txt")
			print 'Noise data Extracted'

			#scipy.optimize.fmin_l_bfgs_b
			if mode!=1: 
				if mode!=4:									#not only linguistic features
					print "Estimating beta"
					
					initialbeta = np.copy(beta)
					initialc = c1
					initialbeta = np.append(initialbeta,initialc)
					solvedbeta = scopt.fmin_l_bfgs_b(objectiveFunction, x0=initialbeta, args=(triad,qyij,polarity),  approx_grad=True)[0]
					beta = solvedbeta[:-1]
					c1 = solvedbeta[-1]
					print 'beta and c evaluated and updated'

				if mode!=2:
					print "Estimating alpha"
					initialalpha = np.copy(alpha)
					initialalphac = alphac
					initialalpha = np.append(initialalpha,initialalphac)
					solvedalpha = scopt.fmin_l_bfgs_b(objectiveFunctionAlpha, x0=initialalpha, args=(edges,qyij,polarity,commonFriendsDict),  approx_grad=True)[0]
					alpha = solvedalpha[:-1]
					alphac = solvedalpha[-1]
					print 'alpha and c evaluated and updated'
				
			print 'evaluating log likelihood to see that it is going in the right direction'
			#This is the log likelihood of the probability of the data given beta. 
			#Here it replaces the log of the probability of the data with the summation of qs and beta 
			#and the normalisation function by c which it learns in every iteration. 
			#It does the same for every structural feature, like common friends
			ll = 0.0
			logpxy = 0.0
			qlogqv = 0.0
			normalizellv = 0.0
			zbeta = 0.0
			zalpha = 0.0

			if mode!=5: 												#not only linguistic features
				logpxy = evaluatelogpxy(qyij,theta,nc,outputVector)

			qlogqv =  qlogq(qyij,nc)
			normalizellv = normalizell(qyij,nc)
			if mode!=1: 
				if mode!=4:												#not only structure features
					zbeta = lllnpmx(beta,c1,triad,qyij) 	
				if mode!=2:					
					zalpha = lllnpmxalpha(alpha,alphac,edges,qyij,commonFriendsDict)
			#zalphadegree = lllnpmxalphadegree(alphadegree,alphadegreec,edges,qyij,characterDegreeDict,maxDegreeDict)
			print 'expected log likelihood = ',(logpxy+zbeta+zalpha-qlogqv-normalizellv)
			loglikelihoods.append(float(logpxy+zbeta+zalpha-qlogqv-normalizellv))
			print 'existing log likelihoods\n',loglikelihoods
			if len(loglikelihoods)>2:
				if(abs(loglikelihoods[-1] - loglikelihoods[-2])<0.01):
					break
		print "Code has run successfully and the parameters are learnt. Run finalClusters.py to get the final clusters."
		break 								
	except:
		print "Looks like there has been an exception because of the initial parameter values. Do you want to reinitialise the parameters are try again?"
		


betaFile= open("betaPickle1.txt","wb")
pickle.dump(beta,betaFile)
betaFile.close()

thetaFile = open("thetaPickle1.txt","wb")
pickle.dump(theta,thetaFile)
thetaFile.close()

qFile = open("qPickle1.txt","wb")
pickle.dump(qyij,qFile)
qFile.close()

alphaFile = open("alphaPickle1.txt","wb")
pickle.dump(alpha,alphaFile)
alphaFile.close()

alphaDegreeFile = open("alphaDegreePickle1.txt","wb")
pickle.dump(alphadegree,alphaDegreeFile)
alphaDegreeFile.close()

loglikelihoodsFile = open("loglikelihoodsPickle.txt","wb")
pickle.dump(loglikelihoods,loglikelihoodsFile)
loglikelihoodsFile.close()

regularisationsFile = open("regularisationParameters.txt","wb")
pickle.dump([thetaR,betaR,alphaR,threshold],regularisationsFile)
regularisationsFile.close()

modeFile = open("mode.txt","wb")
pickle.dump(mode,modeFile)
modeFile.close()