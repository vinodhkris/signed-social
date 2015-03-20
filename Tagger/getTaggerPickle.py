from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import util
import nltk.classify

featuresSet = util.unpickle("featuresSet_best.txt")
classifier = nltk.classify.SklearnClassifier(LinearSVC())
classifier.train(featuresSet)

util.writePickle(classifier,"trained_tagger.pickle")