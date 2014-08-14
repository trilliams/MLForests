MLForests
=========
Without preprocessing of data or post-pruning, currently scores 78.469% accuracy for Titanic survival data on Kaggle: http://www.kaggle.com/c/titanic-gettingStarted

Currently working for numerical and categorical variables in python. To build a tree, run dtrees.py and mlforests.py and then use treebuilder(), which returns a Tree object. For inputs, treebuilder takes a sample in the form of a Pandas DataFrame, a list of the columns you want to train on as strings, and a string of the name of target column. Run treescore(Tree,sample,target) to determine the accuracy of the tree on a target sample. Training accuracy will print when you run treebuilder, but can also be computed accordingly.

To build a forest of trees, use forestbuilder() with the same arguments as treebuilder() and an optional argument for the number of bootstrapped trees, N (defaulted to 1000). This returns a list of trees. Then use forestclassify(forest,testdata) to return a list of classifications for your data set.

Soon to come: 
  -post-pruning capabilities based on information gain for use with validation sets  
  -probabilistic classification for data points with missing values  
  -forestbuilder() for random forest classification by decision tree bagging  


Summary:  
MLForests is a work-in-progress with the goal of being a hard-coded program for creating random forests from a Pandas dataframe. It was created to be both an exercise to refine my Python skills and a more customizable decision tree interface for statistical classification. It will be continuously updated, with the end goal of being an error-robust classifying algorithm for use in data science.

dtrees is an object class containing Tree and Node for building a decision tree, as well as Bootstrap, while mlforests contains the functions necessary to bootstrap a sample from a Pandas dataframe and construct a random forest classifier from it.


