import pandas as pd
import numpy as np
from random import choice
#import matplotlib.pyplot as plt
import math

def bootstrapmean(sample,trials=1000,samplesize=10):
    from numpy import mean
    values = []
    for i in range(trials):
        bootstrapsample = [choice(sample) for i in range(samplesize)]
        values.append(np.mean(bootstrapsample))
    return values

def bootstrapmedian(sample,trials=1000,samplesize=10):
    from numpy import median
    values = []
    for i in range(trials):
        bootstrapsample = [choice(sample) for i in range(samplesize)]
        values.append(np.median(bootstrapsample))
    return values

def entropy(sample,attribute,labels,target):
    totalentropy=0.0
    #implement fractional entropy, expected value
    setcard = float(len(sample[attribute].dropna()))
    targetvals = set(sample[target].dropna())
    for label in labels:
        currentset = sample[sample[attribute]==label]
        pcard = float(len(currentset))
        pentropy = 0.0
        for val in targetvals:
            qcard = float(len(currentset[currentset[target]==val]))
        #THIS WILL NOT WORK, IT IS ONLY RIGHT IN PRINCIPLE
        #MIGHT WORK NOW
            if qcard == 0:
                pentropy -= 0
            else:
                pentropy -= qcard/pcard*math.log(qcard/pcard,2)
        totalentropy += pcard/setcard*pentropy
        #uncomment below for debugging
        #print totalentropy
    return totalentropy

def score(guess,truth):
    right = 0
    total = len(truth)
    for i in range(total):
        if guess[i] == truth[i]:
            right += 1
    return float(right)/total

def entrop(sample,target):
    from math import log
    total = 0.0
    targetvals = set(sample[target])
    card = len(sample)
    for val in targetvals:
        matches = sample[sample[target]==val]
        p = float(len(matches))
        if p == 0:
           total += 0
        else:
            total += -p*math.log(p,2)

def numentropy(sample,attribute,split,target):
    sub = sample.dropna(subset=[attribute])
    under = sub[sub[attribute]<=split]
    over = sub[sub[attribute]>split]
    targetvals = set(sample[target])
    card = len(sample[target])
    #print split

    #compute entropy for samples under the split value
    undertotal = 0.0
    for val in targetvals:
        matches = under[under[target]==val]
        p = float(len(matches))
        if p == 0:
            undertotal += 0
        else:
            undertotal += -(p/card)*math.log((p/card),2)
            #print undertotal

    #compute entropy for samples over the split value
    overtotal= 0.0
    for val in targetvals:
        matches = over[over[target]==val]
        p = float(len(matches))
        if p == 0:
            overtotal += 0
        else:
            overtotal += -(p/card)*math.log((p/card),2)
            #print overtotal

    #combine entropies to get total entropy, want expected value
    overprop = float(len(over))/card
    underprop = float(len(under))/card
    total = overprop*overtotal + underprop*undertotal
    #prints below are for debugging
    #print total
    #print ""
    return total
    
def binumsplit(sample, attribute, target):
    #add a number split for the case of a binary variable
    #This only allows splitting of number continuums on one value
    #May develop trinumericsplit later or perhaps a general numeric split
    #Idea for trinumeric: iteratively move through options, only checking
    #higher options at each point.
    #Just evaluate on unique values
    targetvals = target
    values = sample[attribute].dropna()
    numbers = list(set(values))
    numbers.sort()
    #Want the midpoints of each of these values
    splits = [(i,float(numbers[i] + numbers[i+1])/2) \
              for i in range(len(numbers)-1)]
    bestsplit = None
    minentropy = 1
    for i,split in splits:
        total = numentropy(sample, attribute, split, target)
        if total <= minentropy:
            bestsplit = split
            minentropy = total
    return bestsplit, minentropy

def nodebuilder(sample,attributes,target):
    if len(attributes) == 0:
        #probabilistically assign target value as leaf classifier
        label = problabel(sample,target)
        return Node(target,label)
    elif len(set(sample[target])) == 1:
        #return leaf with classifier as this target value
        label = sample[target].index[0]
        return Node(target,label)
    else:
        #time to build a node the old fashioned way
        #minentropy is unused
        bestattribute,minentropy,value = igfinder(sample,attributes,target)
        if value != None:
            #return (bestattribute,value,True)
            return Node(bestattribute,value,True)
        else:
            return Node(bestattribute,list(set(sample[bestattribute].dropna())))

def noderecurse(Tree,Node,sample,attributes,target):
    #should this be treebuilder?
    #Given a starting node and data set, this will build a tree.
##  if Node.leaf:
##      #useful for debugging
##      print 'echo'
    if not Node.leaf:
        newattributes = attributes[attributes!=Node.attribute]
        attribute,children = Tree.nodes[Node.index]
        if Node.num:
            i,label = zip(*children)
            x = nodebuilder(sample[sample[Node.attribute]<=label[0]],\
                                    newattributes,target)
            Tree.addnode(x,i[0])
            noderecurse(Tree,x,sample[sample[Node.attribute]<=label[0]],newattributes,target)
            y = nodebuilder(sample[sample[Node.attribute]>label[1]],\
                                    newattributes,target)
                #print x.attribute,x.children,x.index
            Tree.addnode(y,i[1])
            noderecurse(Tree,y,sample[sample[Node.attribute]>label[1]],newattributes,target)
                #print x.index,Tree.nodes[x.index]
        else:
            for i,label in children:
                x = nodebuilder(sample[sample[Node.attribute]==label],\
                                    newattributes,target)
                #print x.attribute,x.children,x.index
                Tree.addnode(x,i)
                #print x.index,Tree.nodes[x.index]
                noderecurse(Tree,x,sample[sample[Node.attribute]==label],newattributes,target)

def treebuilder(sample,attributes,target,feedback=True):
    from pandas import Series
    #Will be the general interface for constructing a decision tree
    t = Tree()
    attributes = Series(attributes)
    root = nodebuilder(sample,attributes,target)
    #assuming that the root won't be a leaf
    #that would signify a useless model
    t.addnode(root,root.index)
    #attribute,labels = t.nodes[0]
    #need to keep building labels until every path ends in a leaf
    #noderecurse(t,root,sample,\
    #            attributes[attributes!=root.attribute],target)
    noderecurse(t,root,sample,attributes,target)
    if feedback:
        print 'List of tree nodes with their attribute and branches:'
        print '(A single entry indicates a classifying node and its value)'
        for i in t.nodes.keys():
            print '%i: %s' % (i,t.nodes[i])
        #trainscore = treescore(t,sample,target)
        #print 'Tree correctly predicts %.4f of training set.' % trainscore
    return t

def treescore(Tree,sample,target):
    x=sample.dropna(subset=Tree.attributes)
    #after dropping, x, still preserves it's old indices. Have to reset these.
    #x=titanic
    x.index = range(len(x))
    answers=[]
    solutions=[]
    for i in x.index:
        answers.append(Tree.classify(x.ix[i]))
        solutions.append(x[target].ix[i])
    return score(answers,solutions)

def forestbuilder(sample,attributes,target,N=1000):
    #Builds a random forest using a the treebuilder algorithm
    #First step, Bootstrap N samples of size n
    #1.5: Divide bootstrapped samples into train and test sets on factor a
    from random import choice
    n = len(sample)
    boots = [[sample.ix[choice(sample.index)] for i in range(n)]\
             for j in range(N)]
    bootdfs = [pd.DataFrame(boots[i]) for i in range(N)]
    #Second step, create a decision tree based off of each resample
    forest = [treebuilder(bootdfs[i],attributes,target,False) for i in range(N)]
    #Third step, classify each sample for each tree
    return forest

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


def postprune(Tree,validationsample):
    #time to go over the new sample and fight overfitting
    #simplest implementation: delete anything that adds no new information
    print "This doesn't work yet"


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
    bestattribute = attributes.index[0]
    value = None
    for i,attribute in enumerate(attributes):
        #if the variable is not a binary, small group, or
        #a list of strings, we will need to find a number to split on
        if type(sample[attribute][sample[attribute].index[0]])!= str\
           and len(set(sample[attribute].dropna())) > 9:
            newvalue,newentropy = binumsplit(sample,\
                                          attribute,target)
        #otherwise, we can just use the given methodology
        else:
            newentropy = entropy(sample,attribute,\
                                 set(sample[attribute].dropna()),target)
            newvalue = None
        if newentropy < minentropy:
            bestattribute = attribute
            minentropy = newentropy
            value = newvalue
    return bestattribute,minentropy,value
    #add the numeric igfinder

            
    
    
