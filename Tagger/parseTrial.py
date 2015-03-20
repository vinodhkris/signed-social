import jsonrpc
from simplejson import loads
class Tree:
	def __init__(self,val):
		self.parent = None
		self.children = []
		self.value = val
	def getParent(self):
		return self.parent
	def getChildren(self):
		return self.children
	def getValue(self):
		return self.value

server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))

def getParentSiblings(result,word1):
	root = Tree('')
	initialroot = root
	if 'sentences' not in result:
		return []
	treeroot = None
	w1 = word1
	for word in result['sentences'][0]['parsetree'].split():
		if '(' in word:
			root1 = Tree(word.split('(')[1])
			root1.parent = root
			root.children.append(root1)
			root = root1
		if ')' in word:
			s = 0
			for ch in word:
				if ch == ')':
					s+=1
			if word.split(")")[0].lower() == w1:
				treeroot = root
			for i in xrange(s):
				root = root.parent

	if treeroot == None:
		return []
	siblings = []
	for i in treeroot.parent.children:
		if i == treeroot:
			continue
		siblings.append(i.value)
	val = ""
	r1 = treeroot.parent
	while r1.value!='ROOT' and r1.value=='NP':
		r1 = r1.parent
		if r1.value == '':
			break
	return [r1.value,siblings]

def getDependencies(result,word1):
	x = []
	if 'sentences' not in result:
		return x
	for i in xrange(len(result['sentences'][0]['dependencies'])):
		for word in result['sentences'][0]['dependencies'][i]:
			if word.lower() == word1:
				x.append(result['sentences'][0]['dependencies'][i][0])
	return x