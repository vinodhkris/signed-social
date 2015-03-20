#code to tag words based on BILOU
from sklearn.svm import LinearSVC
import util
import nltk.classify
import nltk
from nltk import word_tokenize,sent_tokenize
import parseTrial
from collections import defaultdict
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import jsonrpc
from simplejson import loads

classifier = util.unpickle("trained_tagger.pickle")

server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))


def initializeModules():
	ps = PorterStemmer()
	pattern = re.compile("[?!.-;:]+")
	commaPattern = re.compile("[,]+")
	return [ps,pattern,commaPattern]

#Get the tokenized form of the sentence, the postags of the sentence and the parse tree
def getSentenceFeatures(line):
	line = line.strip()
	wordTokenized = word_tokenize(line)
	posTags = util.runPOS(line)
	result = {}
	try:
		result = loads(server.parse(line))
	except:
		pass
	return [wordTokenized,posTags,result]

#List of features : Before word, after word, 2 words before, 2 words after, if , appears before, if , appears after, if you or its derivative appears before or after, first non nnp parent, siblings, pos tag before, pos tag after, current pos tag, dependency relationships, main verb
def getFeatures(word,wordTokenized,posTags,parseTree,modules):			
	ps,pattern,commaPattern = modules

	pos = word[1] 									#Position of the word in the sentence
	word = word[0] 									#The word itself 
	
	features = defaultdict(int)
	result = {}
	firstTime = True

	if pos>0:
		features[('beforeWord',ps.stem(wordTokenized[pos-1].lower()))] = 1
		features[('beforePOSTag',posTags[pos-1][1])] = 1
		features[('beforeWordCase',wordTokenized[pos-1].istitle())] = 1
		features[('beforeWordUpperCase',wordTokenized[pos-1].isupper())] = 1
	else:
		features[('beforeWord','')] = 1
		features[('beforePOSTag','')] = 1

	if pos<len(wordTokenized)-1:
		features[('afterWord',ps.stem(wordTokenized[pos+1].lower()))] = 1
		features[('afterPOSTag',posTags[pos+1][1])] = 1
		features[('afterWordCase',wordTokenized[pos+1].istitle())] = 1
		features[('afterWordUpperCase',wordTokenized[pos+1].isupper())] = 1
	else:
		features[('afterWord','')] = 1
		features[('afterPOSTag','')] = 1
	if pos>1:
		features[('before2Words',ps.stem(wordTokenized[pos-2].lower()))] = 1
		features[('before2POSTags',posTags[pos-2][1])] = 1
		features[('before2WordCase',wordTokenized[pos-2].istitle())] = 1
		features[('before2WordUpperCase',wordTokenized[pos-2].isupper())] = 1
	else:
		features[('before2Words','')] = 1
		features[('before2POSTags','')] = 1

	if pos<len(wordTokenized)-2:
		features[('after2Words',ps.stem(wordTokenized[pos+2].lower()))] = 1
		features[('after2POSTags',posTags[pos+2][1])] = 1
		features[('after2WordCase',wordTokenized[pos+2].istitle())] = 1
		features[('after2WordUpperCase',wordTokenized[pos+2].isupper())] = 1
	else:
		features[('after2Words','')] = 1
		features[('after2POSTags','')] = 1

	
	for i in xrange(pos):
		if wordTokenized[i] == ',':
			features['cb'] = 1
		if ps.stem(wordTokenized[i].lower()).replace("'","") == 'you' or ps.stem(wordTokenized[i].lower()).replace("'","") == 'your':
			features['ub'] = 1
		if pattern.match(wordTokenized[i])!=None:
			features['punct_b'] = 1
	for i in xrange(pos+1,len(wordTokenized)):
		if wordTokenized[i] == ',':
			features['ca'] = 1
		if ps.stem(wordTokenized[i].lower()).replace("'","") == 'you' or ps.stem(wordTokenized[i].lower()).replace("'","") == 'your':
			features['ua'] = 1
		if pattern.match(wordTokenized[i])!=None:
			features['punct_a'] = 1

	p = parseTrial.getParentSiblings(parseTree,word.lower())
	if len(p) >0: 
		features[('nonNPparent',p[0])] = 1
		for d in xrange(len(p[1])):
			features[('sibling',p[1][d])] = 1
	x = parseTrial.getDependencies(parseTree,word.lower())
	for dep in x:
		features[('dep',dep)] = 1
	features[('positionFromStart',pos)]=1
	features[('positionFromEnd',len(wordTokenized)-pos)] = 1
	features[('currWord',ps.stem(word.lower()))] = 1
	features[('currPOSTag',posTags[pos][1])] = 1
	features[('currWordCase',word.istitle())] = 1
	return features


if __name__ == '__main__':
	m = initializeModules()
	for sentence in open("sample_text.txt","r"):
		for line in sent_tokenize(sentence):
			line = line.strip()
			out = getSentenceFeatures(line)
			wordTokenized = out[0]
			posTags = out[1]
			parseTree = out[2]
			for i in xrange(len(wordTokenized)):
				word = (wordTokenized[i],i)
				feats = getFeatures(word,wordTokenized,posTags,parseTree,m)
				tag = classifier.classify(feats)
				print word,tag