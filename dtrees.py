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
        #dictionary for weights at each node
        self.counts = {}
        #dictionary for locations of each node
        self.locations = {}
        #dictionary of labels for if a node becomes a leaf
        self.problabels = {}
        #list of attributes that the tree splits on
        self.attributes = []

    def addnode(self, Node, index):
        Node.index = index
        #basic accommodation for dealing with NA attributes
        if Node.attribute not in self.attributes:
            self.attributes.append(Node.attribute)
        self.n += 1
        self.newnodes = []
        self.newlocs = []
        self.num[Node.index]=Node.num
        self.counts[Node.index]=Node.counts
        self.problabels[Node.index]=Node.problabel
        #if a node is a leaf, we store its classifier value
        if Node.leaf:
            self.nodes[index] = Node.children
            self.leaves[index] = True
        else:
            #if a node has children, we assign the new locations
            for (i,label) in Node.children:
                self.newnodes.append((self.n, label))
                self.newlocs.append(self.n)
                self.n += 1
            self.nodes[index] = [Node.attribute, self.newnodes]
            self.locations[index] = self.newlocs
            self.leaves[index] = False


    def classify(self, x):
        node = 0
        while type(self.nodes[node]) == list:
            origin = node
            [attribute, labels] = self.nodes[node]
            if self.num[node]:
                #only built for binary splits, will have to generalize if bigger
                locations, splits = zip(*labels)
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
        #afterwards, enter the label of the final node
        return self.nodes[node]


    def prune(self, index):
        #Recursive tree pruner, prunes node and all nodes below it
        if not self.leaves[index]:
            attribute, labels = self.nodes[index]
            for (i, label) in labels:
                self.prune(i)
            self.leaves[index] = True
            self.nodes[index] = self.problabels[index]
    
    def __len__(self):
        #count nodes
        return len(self.nodes)

class Node:
    def __init__(self, attribute, labels, prob, counts=None, numeric=False, index=0):
        self.index = index
        self.attribute = attribute
        self.leaf = False
        self.num  = numeric
        self.counts = counts
        self.problabel = prob
        if self.num:
            #currently only using binumeric split
            self.children = [(0, labels), (1, labels)]
        else:
            if type(labels) != list:
                self.leaf = True
                self.children = labels
            else:
                self.children = [i for i in enumerate(labels)]

    def __len__(self):
        #count children if it's not a leaf
        return (not self.leaf)*len(self.children)
