import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def bootstrapmean(sample,trials=1000,samplesize=10):
    from numpy import mean
    values = []
    for i in range(trials):
        bootstrapsample = [random.choice(sample) for i in range(samplesize)]
        values.append(np.mean(bootstrapsample))
    return values

def bootstrapmedian(sample,trials=1000,samplesize=10):
    from numpy import median
    values = []
    for i in range(trials):
        bootstrapsample = [random.choice(sample) for i in range(samplesize)]
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
        print totalentropy
    return totalentropy

def score(guess,truth):
    right = 0
    total = len(truth)
    for i in range(total):
        if guess[i] == truth[i]:
            right += 1
    return float(right)/total

def binumsplit(sample, values, target):
    #This only allows splitting of number continuums on one value
    #May develop trinumericsplit later or perhaps a general numeric split
    #Idea for trinumeric: iteratively move through options, only checking
    #higher options at each point.
    #Just evaluate on unique values
    numbers = list(set(values))
    numbers.sort()
    #Want the midpoints of each of these values
    choices = [(i,float(numbers[i] + numbers[i+1])/2) for i in range(len(numbers)-1)]
    index = -1
    minentropy = 1
    for i,number in choices:
        p = float(len(filter(lambda x: x <= number and x != None,values)))
        q = len(values)-p
        if p == 0:
            total = -q*math.log(q,2)
        elif q == 0:
            total = -p*math.log(p,2)
        else:
            total = -p*math.log(p,2)-q*log(q,2)
        if total <= minentropy:
            index = number
            minentropy = total
    return number,entropy
        #compute entropy of each of these splits
        #take advantage of data series and important attribute
        #first divide by current label values, then by target
        #classifier
        #need variable for set of label names
        #need variable for set of target values
        #for now should just consider binary targets
        #keep creating nodes until the size of the set
        #of target values for the remaining group is 1
        #or the size of the remaining attributes is zero
        #if you run out of attributes, assign the final value
        #probabilistically
        #look for len(set(sampletarget))

def nodebuilder(sample,attributes,target):
    if len(attributes) == 0:
        #probabilistically assign target value as leaf classifier
        label = problabel(sample,target)
        return Node(target,label)
    elif len(set(sample[target])) == 1:
        #return leaf with classifier as this target value
        label = sample[target][0]
        return Node(target,label)
    else:
        #time to build a node the old fashioned way
        bestattribute,minentropy,value = igfinder(sample,attributes,target)
        #not currently using entropy
        if value!=None:
            return Node(bestattribute,value,True)
        else:
            return Node(bestattribute,list(set(sample[bestattribute].dropna())))

def noderecurse(Tree,Node,sample,attributes,target):
    #should this be treebuilder?
    if not Node.leaf:
        newattributes = attributes[attributes!=Node.attribute]
        for i,label in Node.children:
            x = nodebuilder(sample[sample[Node.attribute]==label],\
                                newattributes,target)
            Tree.addnode(x,Node.index)
            noderecurse(Tree,x,sample[sample[Node.attribute]==label],newattributes,target)


def treebuilder(sample,attributes,target):
    from pandas import Series
    t = Tree()
    attributes = Series(attributes)
    root = nodebuilder(sample,attributes,target)
    #assuming that the root won't be a leaf
    #that would signify a useless model
    t.addnode(root,0)
    attribute,labels = t.nodes[0]
    #need to keep building labels until every path ends in a leaf
    for i,label in labels:
        noderecurse(t,root,sample[sample[root.attribute]==label],\
                    attributes[attributes!=root.attribute],target)
    return t            
    

def postprune(tree,validationsample):
    #time to go over the new sample and fight overfitting
    #simplest implementation: delete anything that adds no new information
    print "This doesn't work yet"


def problabel(sample,target):
    maximum = -1
    maxlabel = ''
    for label in set(target):
        total = 0
        for i in sample[target]:
            total += (i==label)
        if total >= maximum:
            maximum = total
            maxlabel = label
    return maxlabel

def igfinder(sample,attributes,target):
    #minindex = -1
    minentropy = 1
    #placeholder
    bestattribute = attributes.index[0]
    value = None
    for i,attribute in enumerate(attributes):
        if type(sample[attribute][sample[attribute].index[0]])!= str:
            newentropy,newvalue = binumsplit(sample,\
                                          sample[attribute].dropna(),target)
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

            
    
    
