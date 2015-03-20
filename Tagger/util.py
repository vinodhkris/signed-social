import nltk
import urllib2
import simplejson
import zlib
import time
import cPickle as pickle 

def unpickle(filename):
	f = open(filename,"rb") 
	data = pickle.load(f)
	return data

def writePickle(struct, filename):
	file1 = open(filename,"wb") 			
	pickle.dump(struct,file1)
	file1.close()

def prepareSentence(sample):
	sentences = nltk.sent_tokenize(sample)
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	chunked_sentences = nltk.batch_ne_chunk(tagged_sentences, binary=True)
	return chunked_sentences
 
def extract_entity_names(t):
	entity_names = []
	if hasattr(t, 'node') and t.node:
		if t.node == 'NE':
			entity_names.append(' '.join([child[0] for child in t]))
		else:
			for child in t:
				entity_names.extend(extract_entity_names(child))
	return entity_names
 
def runNER(line):
	entity_names = []
	chunkedSentences = prepareSentence(line)
	for tree in chunkedSentences:
		entity_names.extend(extract_entity_names(tree))

	return set(entity_names)

def runPOS(line):
	return nltk.pos_tag(nltk.word_tokenize(line))

def getAddressTerms():
	file1 = open("finalAddressTermsOutput.txt","r")
	addressTerms = []
	for line in file1:
		addressTerms.append(line.rstrip('\'\"-,.:;!?\n ').lower())
	return addressTerms

def getLastName(movie):
	apikey = "annfa3qwr35zuraaneurs99j"
	movie = movie.replace(" ","%20")
	#print movie
	searchurl = "http://api.rottentomatoes.com/api/public/v1.0/movies.json?apikey=annfa3qwr35zuraaneurs99j&q="+movie+"&page_limit=50&page=1"	
	moviesdata = {}
	try:
		urlreq = searchurl
		f = urllib2.urlopen(urlreq)
	except:
		return []
	data = {}
	try:
		f1 = zlib.decompress(f.read(), 16+zlib.MAX_WBITS)
		data = simplejson.loads(f1)
	except zlib.error:
		try:	
			f = urllib2.urlopen(urlreq)
			data = simplejson.load(f)
		except:
			return []
	try:
		movieId = data["movies"][0]["id"]
		castURL = "http://api.rottentomatoes.com/api/public/v1.0/movies/"+movieId+"/cast.json?apikey=annfa3qwr35zuraaneurs99j"
	except:
		return []
	try:
		urlreq = castURL
		f = urllib2.urlopen(urlreq)
	except:
		return []
	castdata = {}
	try:
		f1 = zlib.decompress(f.read(), 16+zlib.MAX_WBITS)
		castdata = simplejson.loads(f1)
	except zlib.error:
		try:
			f = urllib2.urlopen(urlreq)
			castdata = simplejson.load(f)
		except:
			return []
	try:
		charactersList = []
		for num in xrange(len(castdata["cast"])):
			for num1 in xrange(len(castdata["cast"][num]["characters"])):
				if "/" in castdata["cast"][num]["characters"][num1]:
					charactersList.append(castdata["cast"][num]["characters"][num1].split("/")[0].lower())
				else:
					charactersList.append(castdata["cast"][num]["characters"][num1].lower())
	except:
		return []
	#print charactersList
	return charactersList