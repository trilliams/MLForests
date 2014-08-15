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
        self.counts = {}
        self.locations = {}
        self.attributes = []

    def addnode(self,Node,index):
        Node.index = index
        #basic accommodation for dealing with NA attributes
        if Node.attribute not in self.attributes:
            self.attributes.append(Node.attribute)
        self.n += 1
        self.newnodes = []
        self.newlocs = []
        self.num[Node.index]=Node.num
        self.counts[Node.index]=Node.counts
        if Node.leaf:
            self.nodes[index] = Node.children
            self.leaves[index] = True
        else:
            for (i,label) in Node.children:
                self.newnodes.append((self.n,label))
                self.newlocs.append(self.n)
                self.n += 1
            self.nodes[index] = [Node.attribute,self.newnodes]
            self.locations[index] = self.newlocs
            self.leaves[index] = False


    def classify(self,x):
        node = 0
        while type(self.nodes[node]) == list:
            origin = node
            [attribute,labels] = self.nodes[node]
            if self.num[node]:
                #only built for binary splits, will have to generalize
                locations,splits = zip(*labels)
                if np.isnan(x[attribute]):
                    randindex = countchoice(self.counts[node])
                    node = self.locations[node][randindex]                
                else:
                    if x[attribute] <= splits[0]:
                        node = locations[0]
                    else:
                        node = locations[1]
            else:
                for (location,label) in labels:
                    if x[attribute] == label:
                        node = location
            #if the node hasn't matched, send it down a branch
            #probabilistically
            if node == origin:
                randindex = countchoice(self.counts[node])
                node = self.locations[node][randindex]
            
        return self.nodes[node]


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
    def __init__(self,attribute,labels,counts=None,numeric=False,index=0):
        self.index = index
        self.attribute = attribute
        self.leaf = False
        self.num  = numeric
        self.counts = counts
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
        #count children if it's not a leaf
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


        

    
            
