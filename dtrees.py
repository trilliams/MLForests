from random import choice
from numpy import mean,median

class Tree:
    def __init__(self):
        self.n = 0
        #dictionary for node attributes
        self.nodes = {}
        #dictionary for finding leaves
        self.leaves = {}
        #dictionary for determining numeric classifiers
        self.num = {}
        self.attributes = []

    def addnode(self,Node,index=n):
        Node.index = index
        #basic accommodation for dealing with NA attributes
        if Node.attribute not in self.attributes:
            self.attributes.append(Node.attribute)
        self.n += 1
        self.newnodes = []
        self.num[Node.index]=Node.num
        if Node.leaf:
            self.nodes[index] = Node.children
            self.leaves[index] = True
        else:
            for (i,label) in Node.children:
                self.newnodes.append((self.n,label))
                self.n += 1
            self.nodes[index] = [Node.attribute,self.newnodes]
            self.leaves[index] = False
            #add the numeric node changer

    def classify(self,x):
        node = 0
        while type(self.nodes[node]) == list:
            origin = node
            [attribute,labels] = self.nodes[node]
            if self.num[node]:
                #only built for binary splits, will have to generalize
                locations,splits = zip(*labels)
                if x[attribute] <= splits[0]:
                    node = locations[0]
                else:
                    node = locations[1]
            else:
                for (location,label) in labels:
                    if x[attribute] == label:
                        node = location
            if node == origin:
                node = labels[0][0]
            #if it hasn't matched, send it left. quick fix.
            
        return self.nodes[node]
        #add the numeric classifier
    #having a problem where sometimes nodes get into a classification they
    #aren't qualified for and then it breaks the system. While this would
    #be a probabilistic case, I'm not equipped to do that yet

    def prunenode(self,index):
        #This does not work yet
        if not self.leaves[index]:
            attribute,labels = self.nodes[index]
            for (i,label) in labels:
                prunenode(i)
        del self.leaves[index]
        del self.nodes[index]
    
    def __len__(self):
        #count nodes
        return len(self.nodes)

class Node:
    def __init__(self,attribute,labels,numeric=False,index=0):
        self.index = index
        self.attribute = attribute
        self.leaf = False
        self.num  = numeric
        if self.num:
            #currently only using binumeric split
            self.children = [(0,labels),(1,labels)]
        else:
            if type(labels) != list:
                self.leaf = True
                self.children = labels
            else:
                self.children = [i for i in enumerate(labels)]

    def __len__(self):
        #count
        return (not self.leaf)*len(self.children)

class Bootstrap:
    def __init__(self,sample):
        N = 1000
        n = len(sample)
        self.bootstraps = []
        for i in range(N):
            self.bootstraps.append([choice(sample) for i in range(n)])

    def __len__(self):
        return len(self.bootstraps)

    def __repr__(self):
        return '%i bootstraps of %i samples' %\
               (len(self.bootstraps),len(self.bootstraps[0]))

    def size(self):
        print len(self.bootstraps),'bootstraps of',\
              len(self.bootstraps[0]),'samples'

    def mean(self):
        means = [mean(self.bootstraps[i]) for i \
                 in range(len(self.bootstraps))]
        return means
    
    def median(self):
        medians = [mean(self.bootstraps[i]) for \
                   i in range(len(self.bootstraps))]
        return medians


        

    
            
