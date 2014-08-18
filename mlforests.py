import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy

def entropy(sample,attribute,labels,target):
    totalentropy=0.0
    setcard = float(len(sample[attribute].dropna()))
    targetvals = set(sample[target].dropna())
    counts = []
    for label in labels:
        currentset = sample[sample[attribute]==label]
        pcard = float(len(currentset))
        counts.append(pcard)
        pentropy = 0.0
        for val in targetvals:
            qcard = float(len(currentset[currentset[target]==val]))
            if qcard == 0:
                pentropy -= 0
            else:
                pentropy -= qcard/pcard*math.log(qcard/pcard,2)
        totalentropy += pcard/setcard*pentropy
        #uncomment below for debugging
        #print totalentropy
    return totalentropy,counts

def score(guess,truth):
    right = 0
    total = len(truth)
    for i in range(total):
        if guess[i] == truth[i]:
            right += 1
    return float(right)/total

def numentropy(sample,attribute,split,target):
    ##computes entropy for splits on numeric values
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
    
def binumsplit(sample, attribute, target):
    ##Finds the best value to split a numeric column on.
    ##Just evaluates on unique values.
    targetvals = target
    values = sample[attribute].dropna()
    numbers = list(set(values))
    numbers.sort()
    counts = 0

    #Want the midpoints of each of these values
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
    return bestsplit, minentropy,  counts

def nodebuilder(sample,attributes,target):
    ##Builds a node from current sample, attributes, and target.
    prob = problabel(sample,target)
    if len(attributes) == 0:
        #probabilistically assign target value as leaf classifier
        label = prob
        return Node(target,label,prob)
    elif len(set(sample[target])) == 1:
        #return leaf with classifier as this target value
        label = sample[target].ix[sample[target].index[0]]
        return Node(target,label,prob)
    else:
        #time to build a node the old fashioned way
        #minentropy is unused
        bestattribute,minentropy,value,counts = \
                                            igfinder(sample,attributes,target)
        if value != None:
            #return (bestattribute,value,True)
            return Node(bestattribute,value,prob,counts,True)
        else:
            return Node(bestattribute,\
                        list(set(sample[bestattribute].dropna())),prob,counts)

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

def noderecurse(Tree,Node,sample,attributes,target):
    #Given a starting node and data set, this will build a tree.
    if not Node.leaf:
        newattributes = attributes[attributes!=Node.attribute]
        attribute,children = Tree.nodes[Node.index]
        if Node.num:
            under,over = numnancleaner(Node,sample)
            i,label = zip(*children)
            x = nodebuilder(under,newattributes,target)
            Tree.addnode(x,i[0])
            noderecurse(Tree,x,under,newattributes,target)
            y = nodebuilder(over,newattributes,target)
                #print x.attribute,x.children,x.index
            Tree.addnode(y,i[1])
            noderecurse(Tree,y,over,newattributes,target)
                #print x.index,Tree.nodes[x.index]
        else:
            listnancleaner(Node,sample)
            for i,label in children:
                x = nodebuilder(sample[sample[Node.attribute]==label],\
                                    newattributes,target)
                #print x.attribute,x.children,x.index
                Tree.addnode(x,i)
                #print x.index,Tree.nodes[x.index]
                noderecurse(Tree,x,sample[sample[Node.attribute]==label],newattributes,target)

def treebuilder(sample,attributes,target,feedback=True):
    from pandas import Series
    #General interface for constructing a decision tree
    t = Tree()
    attributes = Series(attributes)
    root = nodebuilder(sample,attributes,target)
    #assuming that the root won't be a leaf
    #that would signify a useless model
    t.addnode(root,root.index)
    #need to keep building labels until every path ends in a leaf
    noderecurse(t,root,sample,attributes,target)
    if feedback:
        print 'List of tree nodes with their attribute and branches:'
        print '(A single entry indicates a classifying node and its value)'
        for i in t.nodes.keys():
            print '%i: %s' % (i,t.nodes[i])
        trainscore = treescore(t,sample,target)
        print 'Tree correctly predicts %.4f of training set.' % trainscore
    return t

def treescore(Tree,sample,target):
    x = sample
    x.index = range(len(x))
    answers=[]
    solutions=[]
    for i in x.index:
        answers.append(Tree.classify(x.ix[i]))
        solutions.append(x[target].ix[i])
    return score(answers,solutions)

def forestbuilder(sample,attributes,target,N=10,alpha=.30,prune=True):
    #Builds a random forest using a the treebuilder algorithm
    #alpha determines what percentage of the bootstrapped sample is
    #used for validation set.
    #if prune is set to false, postpruning and cross-validation are off.
    #1: Bootstrap N samples of size n
    n = len(sample)
    #1.25: Reset index for each bootstrap
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

def forestiter(sample,attributes,target,N=1000):
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
    #after while
    if straps != 0:
        forest = forestbuilder(sample,attributes,target,N=straps)
        for i in range(n):
            guessdict[i] += guesses[i]
        del forest
    results = [modefreq(guessdict[guesses]) for guesses in guessdict]
    return results

def rfclassifier(sample,testsample,attributes,target,N=1000):
    ##iteratively builds a random forest on sample and classifies test sample
    n = len(sample)
    testn = len(testsample)
    straps = N
    guessdict = {}
    for i in range(testn):
        guessdict[i] = []
    #builds ten forests at a time, classifies through them, and then resets
    while straps >= 10:
        straps -= 10
        forest = forestbuilder(sample,attributes,target,N=10)
        guesses = treeiter(forest,testsample)
        for i in range(testn):
            guessdict[i] += guesses[i]
        del forest
    #after while
    if straps != 0:
        forest = forestbuilder(sample,attributes,target,N=straps)
        guesses = treeiter(forest,testsample)
        for i in range(testn):
            guessdict[i] += guesses[i]
        del forest
    results = [modefreq(guessdict[guesses]) for guesses in guessdict]
    return results
    
def treeiter(forest,sample):    
    labels = []
    for i in sample.index:
        indlabels = []
        for tree in forest:
            t = tree.classify(sample.ix[i])
            indlabels.append(t)
        labels.append(indlabels)
    return labels  

def forestclassify(forest,sample):
    results = []
    for i in sample.index:
        indlabels = []
        for tree in forest:
            t = tree.classify(sample.ix[i])
            indlabels.append(t)
        label = modefreq(indlabels)
        results.append(label)
    #Fourth step, generate results and accompanying frequencies
    #results = [modefreq(treelabels) for treelabels in labels]
    return results
    #print 'Forest'

def dfresample(sample):
    from random import choice
#    boots = [sample.ix[choice(

def modefreq(sample):
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


def postprune(Tree,validationsample,target):
    #time to go over the new sample and fight overfitting
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
    #print basescore        


def problabel(sample,target):
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

def igfinder(sample,attributes,target):
    minentropy = 1
    bestattribute = attributes.ix[attributes.index[0]]
    value = None
    counts = (1)
    for i,attribute in enumerate(attributes):
        #if the variable is not a binary, small group, or
        #a list of strings, we will need to find a number to split on
        if type(sample[attribute][sample[attribute].index[0]])!= str:
        #old option below
        #if type(sample[attribute][sample[attribute].index[0]])!= str\
        #   and len(set(sample[attribute].dropna())) > 9:
            newvalue,newentropy,newcounts = binumsplit(sample,\
                                          attribute,target)
            if newentropy < minentropy:
                bestattribute = attribute
                minentropy = newentropy
                counts = newcounts
                value = newvalue            
        #otherwise, we just use the given methodology
        else:
            newentropy,newcounts = entropy(sample,attribute,\
                                 set(sample[attribute].dropna()),target)
            newvalue = None
        if newentropy < minentropy:
            bestattribute = attribute
            minentropy = newentropy
            counts = newcounts
            value = newvalue
    return bestattribute,minentropy,value,counts


def countchoice(counts):
    #temporary fix
    #returns the index of a list according to the weights
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
    
