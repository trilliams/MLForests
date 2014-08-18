import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy

################# RANDOM FOREST CLASSIFIER ##################

def rfclassifier(sample,testsample,attributes,target,\
                 N=1000,alpha=0.30,prune=True,n=False):
    ##Iteratively builds a random forest on sample and classifies test sample
    if not n:
        n = len(sample)
    testn = len(testsample)
    straps = N
    guessdict = {}
    for i in range(testn):
        guessdict[i] = []
    #builds ten forests at a time, classifies through them, and then resets
    while straps >= 10:
        straps -= 10
        forest = forestbuilder(sample,attributes,target,\
                               N=10,alpha=alpha,prune=prune,n=n)
        guesses = treeiter(forest,testsample)
        for i in range(testn):
            guessdict[i] += guesses[i]
        del forest
    #after while
    if straps != 0:
        forest = forestbuilder(sample,attributes,target,\
                               N=straps,alpha=alpha,prune=prune,n=n)
        guesses = treeiter(forest,testsample)
        for i in range(testn):
            guessdict[i] += guesses[i]
        del forest
    results = [modefreq(guessdict[guesses]) for guesses in guessdict]
    return results

################# FOREST FUNCTIONS ##################

def forestbuilder(sample,attributes,target,N=10,alpha=.30,prune=True,n=False):
    ##Builds a random forest using a the treebuilder algorithm
    #alpha determines what percentage of the bootstrapped sample is
    #used for validation set.
    #if prune is set to false, postpruning and cross-validation are off.
    #1: Bootstrap N samples of size n, defaults to length of sample
    if not n:
        n = len(sample)
    #need to reset index for each bootstrap to avoid duplicate indices
    boots = [[sample.ix[random.choice(sample.index)] for i in range(n)]\
             for j in range(N)]
    bootdfs = [pd.DataFrame(boots[i],index=range(n)) for i in range(N)]
    #2a.1: Divide bootstrapped samples into sets for cross-validation
    if prune:
        traindfs = [bootdfs[i][0:int(n*(1-alpha))] for i in range(N)]
        valdfs   = [bootdfs[i][int(n*(1-alpha)+1)::] for i in range(N)]
        #2a.2: create a decision tree based off of each resample
        forest = [treebuilder(traindfs[i],attributes,target,False)\
                  for i in range(N)]
        #2a.3: postprune each tree using the validation data
        for i in range(N):
            postprune(forest[i],valdfs[i],target)
    #2b: create a decision tree based off of each resample
    else:
        forest = [treebuilder(bootdfs[i],attributes,target,False)\
                  for i in range(N)]
    #3: output tree
    return forest

def forestiter(sample,attributes,target,N=1000,n=False):
    ##Iterative forest builder to save memory
    if not n:
        n = len(sample)
    straps = N
    guessdict = {}
    for i in range(n):
        guessdict[i] = []
    #builds ten forests at a time, classifies through them, and then resets
    while straps >= 10:
        straps -= 10
        forest = forestbuilder(sample,attributes,target,N=10)
        guesses = treeiter(forest,sample)
        for i in range(n):
            guessdict[i] += guesses[i]
        del forest
    #after less than ten to go, we make the remainder
    if straps != 0:
        forest = forestbuilder(sample,attributes,target,N=straps)
        for i in range(n):
            guessdict[i] += guesses[i]
        del forest
    #once we have all of our logged guesses, we find their modes
    results = [modefreq(guessdict[guesses]) for guesses in guessdict]
    return results

def forestclassify(forest,sample):
    ##Takes a forest and returns guesses for each point in a sample
    results = []
    for i in sample.index:
        indlabels = []
        for tree in forest:
            t = tree.classify(sample.ix[i])
            indlabels.append(t)
        label = modefreq(indlabels)
        results.append(label)
    return results

################# TREE FUNCTIONS ##################

def treebuilder(sample,attributes,target,feedback=True):
    ##General interface for constructing a decision tree
    t = Tree()
    attributes = pd.Series(attributes)
    root = nodebuilder(sample,attributes,target)
    #assuming that the root won't be a leaf
    #that would signify a useless model
    t.addnode(root,root.index)
    #need to keep building labels until every path ends in a leaf
    noderecurse(t,root,sample,attributes,target)
    #returns basic info about the tree if desired
    if feedback:
        print 'List of tree nodes with their attribute and branches:'
        print '(A single entry indicates a classifying node and its value)'
        for i in t.nodes.keys():
            print '%i: %s' % (i,t.nodes[i])
        trainscore = treescore(t,sample,target)
        print 'Tree correctly predicts %.4f of training set.' % trainscore
    return t

def treeiter(forest,sample):    
    ##Iteratively classifies a sample a point at a time through the forest
    labels = []
    for i in sample.index:
        indlabels = []
        for tree in forest:
            t = tree.classify(sample.ix[i])
            indlabels.append(t)
        labels.append(indlabels)
    return labels

def postprune(Tree,validationsample,target):
    ##Prunes the tree by testing its performance against a validation set
    #goes over the new sample and fight overfitting
    #simplest implementation: delete anything that adds no new information
    basescore = treescore(Tree,validationsample,target)
    while True:
        #Go through leaf nodes from the bottom up
        leaves = [leaf for leaf in Tree.leaves if Tree.leaves[leaf]]
        for leaf in leaves[::-1]:
            Tree2 = deepcopy(Tree)
            Tree2.prune(leaf)
            newscore = treescore(Tree2,validationsample,target)
            if newscore < basescore:
                Tree = Tree2
                basescore = newscore
                break
        #if we go through all leaves in any iteration and don't prune, that's it
        break

def treescore(Tree,sample,target):
    ##Simple function for scoring a tree's predictions against the target value
    x = sample
    x.index = range(len(x))
    answers=[]
    solutions=[]
    for i in x.index:
        answers.append(Tree.classify(x.ix[i]))
        solutions.append(x[target].ix[i])
    return score(answers,solutions)


def score(guess,truth):
    ##General scoring function, tells you the percent of values that are equal
    right = 0
    total = len(truth)
    for i in range(total):
        if guess[i] == truth[i]:
            right += 1
    return float(right)/total

################# NODE-BUILDING FUNCTIONS ##################

def nodebuilder(sample,attributes,target):
    ##Builds a node from current sample, attributes, and target.
    prob = problabel(sample,target)
    #check to see if there are attributes left
    if len(attributes) == 0:
        #probabilistically assign target value as leaf classifier
        label = prob
        return Node(target,label,prob)
    #check to see if all points have the same target value
    elif len(set(sample[target])) == 1:
        #return leaf with classifier as this target value
        label = sample[target].ix[sample[target].index[0]]
        return Node(target,label,prob)
    #after checking the base cases, build a traditional node
    else:
        #find the vitals of the new node through igfinder
        #minentropy is unused here
        bestattribute,minentropy,value,counts = \
                                            igfinder(sample,attributes,target)
        #if igfinder returned a value, we have a binary numeric split
        if value != None:
            return Node(bestattribute,value,prob,counts,True)
        #if it did not, we have a categorical split
        else:
            return Node(bestattribute,\
                        list(set(sample[bestattribute].dropna())),prob,counts)


def noderecurse(Tree,Node,sample,attributes,target):
    ##Given a starting node and data set, this will build a tree.
    #if a node is a leaf, no need to recurse
    if not Node.leaf:
        #remove the attribute 
        newattributes = attributes[attributes!=Node.attribute]
        attribute,children = Tree.nodes[Node.index]
        #if the node is numerically split, we need to make two groups
        if Node.num:
            #get two samples with values under and over the split, respectively
            under,over = numnancleaner(Node,sample)
            i,label = zip(*children)
            #build the samples of values under the split
            x = nodebuilder(under,newattributes,target)
            Tree.addnode(x,i[0])
            noderecurse(Tree,x,under,newattributes,target)
            #build the samples of values over the split
            y = nodebuilder(over,newattributes,target)
            Tree.addnode(y,i[1])
            noderecurse(Tree,y,over,newattributes,target)
        #if the node is categorically split, we need a group for each label    
        else:
            #probabilistically assign labels to points missing the label
            listnancleaner(Node,sample)
            #build a node for each label
            for i,label in children:
                x = nodebuilder(sample[sample[Node.attribute]==label],\
                                    newattributes,target)
                Tree.addnode(x,i)
                noderecurse(Tree,x,sample[sample[Node.attribute]==label],\
                            newattributes,target)

def igfinder(sample,attributes,target):
    ##Function to find the attribute of a sample that will return the
    ##largest information gain by splitting on it
    #initiate a dummy minimum entropy, start with the first attribute
    minentropy = 1
    bestattribute = attributes.ix[attributes.index[0]]
    value = None
    counts = (1)
    for i,attribute in enumerate(attributes):
        #if the attribute is purely numeric and it has a
        #large range of values, we will run numeric split.
        if (type(sample.dtypes[attribute]) == np.float64 \
           or type(sample.dtypes[attribute]) == np.int64) \
           and len(set(sample[attribute].dropna())) > 5:
            newvalue,newentropy,newcounts = \
                                    binumsplit(sample,attribute,target)
            #if this produces a lower current entropy, use these values
            if newentropy < minentropy:
                bestattribute = attribute
                minentropy = newentropy
                counts = newcounts
                value = newvalue            
        #for any other attribute, we run the old methodology
        else:
            newentropy,newcounts = entropy(sample,attribute,\
                                 set(sample[attribute].dropna()),target)
            newvalue = None
        #if this produces a lower current entropy, use these values
        if newentropy < minentropy:
            bestattribute = attribute
            minentropy = newentropy
            counts = newcounts
            value = newvalue
    return bestattribute,minentropy,value,counts

################# ENTROPY FINDING FUNCTIONS ##################

def binumsplit(sample, attribute, target):
    ## Finds the best value to split a numeric column on,
    ## just evaluates on unique values.
    targetvals = target
    values = sample[attribute].dropna()
    numbers = list(set(values))
    numbers.sort()
    counts = 0
    # use the midpoints of each of these values
    splits = [(i,float(numbers[i] + numbers[i+1])/2) \
              for i in range(len(numbers)-1)]
    bestsplit = None
    minentropy = 1
    for i,split in splits:
        total,newcounts = numentropy(sample, attribute, split, target)
        if total <= minentropy:
            bestsplit = split
            minentropy = total
            counts = newcounts
    return bestsplit, minentropy, counts

def numentropy(sample,attribute,split,target):
    ## Computes entropy for splits on numeric values
    sub = sample.dropna(subset=[attribute])
    under = sub[sub[attribute]<=split]
    over = sub[sub[attribute]>split]
    targetvals = set(sample[target])
    card = len(sample[target])
    counts = (len(under),len(over))
    #compute entropy for samples under the split value
    undertotal = 0.0
    for val in targetvals:
        matches = under[under[target]==val]
        p = float(len(matches))
        if p == 0:
            undertotal += 0
        else:
            undertotal += -(p/card)*math.log((p/card),2)    
    #compute entropy for samples over the split value
    overtotal= 0.0
    for val in targetvals:
        matches = over[over[target]==val]
        p = float(len(matches))
        if p == 0:
            overtotal += 0
        else:
            overtotal += -(p/card)*math.log((p/card),2)      
    #combine entropies to get total entropy, want expected value
    overprop = float(len(over))/card
    underprop = float(len(under))/card
    total = overprop*overtotal + underprop*undertotal
    return total,counts

def entropy(sample,attribute,labels,target):
    ##Compute expected entropy for a node split on 'attribute'
    totalentropy=0.0
    setcard = float(len(sample[attribute].dropna()))
    targetvals = set(sample[target].dropna())
    counts = []
    #separate sample by labels
    for label in labels:
        currentset = sample[sample[attribute]==label]
        pcard = float(len(currentset))
        counts.append(pcard)
        pentropy = 0.0
        #compute entropy for each individual branch
        for val in targetvals:
            qcard = float(len(currentset[currentset[target]==val]))
            if qcard == 0:
                pentropy -= 0
            else:
                pentropy -= qcard/pcard*math.log(qcard/pcard,2)
        totalentropy += pcard/setcard*pentropy
    return totalentropy,counts

################# NAN CLEANING FUNCTIONS ##################

def numnancleaner(Node,sample):
    ##Resolves nans for numeric variables
    location,labels = zip(*Node.children)
    split = labels[0]
    weights = Node.counts
    attribute = Node.attribute
    under = sample[sample[attribute] <= split]
    #next line might not be necessary
    under = under.dropna(subset=[attribute])
    over = sample[sample[attribute] > split]
    #next line might not be necessary
    over = over.dropna(subset=[attribute])
    nans = sample[sample[attribute].apply(np.isnan)]
    for index in nans.index:
        choice = countchoice(weights)
        if choice == 0:
            under.loc[index] = nans.loc[index]
        elif choice == 1:
            over.loc[index] = nans.loc[index]
    return under,over
    
def listnancleaner(Node,sample):
    ##Resolves nans for categorical variables
    weights = Node.counts
    location,labels = zip(*Node.children)
    attribute = Node.attribute
    for index in sample.index:
        if sample[attribute][index] not in labels:
            choice = countchoice(weights)
            sample[attribute][index] = labels[choice]

def problabel(sample,target):
    ##Finds the most common target value of a sample
    maximum = -1
    maxlabel = ''
    for label in set(sample[target]):
        total = 0
        for i in sample[target]:
            total += (i==label)
        if total >= maximum:
            maximum = total
            maxlabel = label
    return maxlabel

def countchoice(counts):
    ##Returns the index of a list, chosen according to weights
    if type(counts) == int:
        return 0
    else:
        total = sum(w for w in counts)
        r = random.uniform(0, total)
        upto = 0
        index = 0
        for w in counts:
            upto += w
            if upto > r:
                return index
            index += 1     

################# GENERAL ML TOOLS ##################

def splitframe(dataframe,percentinfirstframe=0.50):
    ##Splits a data frame in two.
    from random import sample
    #Need to make sure percent is a valid number
    if not (percentinfirstframe >=0 and percentinfirstframe <=1):
        assert False, 'Percent must be a number in [0,1]'
    n = len(dataframe)
    trainsize = int(n*percentinfirstframe)
    testsize = n - trainsize
    trains = random.sample(dataframe.index,trainsize)
    tests = [i for i in dataframe.index if i not in trains]
    traindata = [dataframe.ix[i] for i in trains]
    traindf = pd.DataFrame(traindata,index=range(trainsize))
    testdata = [dataframe.ix[i] for i in tests]
    testdf = pd.DataFrame(testdata,index=range(testsize))
    print 'First data frame has %i samples, second has %i.' % (trainsize,testsize)
    return traindf,testdf

def modefreq(sample):
    ##Returns the mode of a sample and the frequency of said mode
    counts = {}
    for i in set(sample):
        counts[i] = sample.count(i)
    keys = counts.keys()
    values = counts.values()
    maxcount = max(values)
    index = values.index(maxcount)
    mode = keys[index]
    freq = float(maxcount)/len(sample)
    return mode,freq     
    
