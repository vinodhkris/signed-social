#########################To write all the full names to the file###################################################
import cPickle as pickle

def unpickle(filename):
	f = open(filename,"rb") 
	heroes = pickle.load(f)
	return heroes

fn = unpickle("fullNames.txt")
f = open("movie_characters_metadata.txt","r")

for line in f:
	entities = line.split(' +++$+++ ')
	movie = entities[2].strip()
	charname = entities[1].strip()
	char1 = fn[movie][charname]
	entities[1] = char1
	print ' +++$+++ '.join(entities),
