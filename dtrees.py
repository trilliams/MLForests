from random import choice
from numpy import mean,median

class Tree:
    def __init__(self):
        self.n = 0
        self.nodes = {}
        self.leaves = {}

    def addnode(self,Node,parentindex):
        Node.index = self.n
        self.newnodes = []
        if Node.leaf:
            self.nodes[Node.index] = Node.children
            self.leaves[Node.index] = True
        else:
            for (i,label) in Node.children:
                self.n += 1
                self.newnodes.append((self.n,label))
            self.nodes[parentindex] = [Node.attribute,self.newnodes]
            self.leaves[parentindex] = False
            #add the numeric node changer

    def classify(self,x):
        node = 0
        while type(self.nodes[node]) == list:
            [attribute,labels] = self.nodes[node]
            for (location,label) in labels:
                if x[attribute] == label:
                    node = location
        return self.nodes[node]
        #add the numeric classifier

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
    def __init__(self,attribute,labels,numeric=False):
        self.index = 0
        self.attribute = attribute
        self.leaf = False
        self.num  = numeric
        if self.num:
            #currently using binumeric split only and -1000,1000 as lower bounds
            self.children = [(-1000,labels),(labels,1000)]
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


        

    
            
