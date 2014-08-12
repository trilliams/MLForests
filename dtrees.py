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
                self.newnodes.append(self.n,label)
                self.n += 1
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
        

    
            
