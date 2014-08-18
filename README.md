MLForests
=========
Now with full classification functionality! MLForests is a random forest classifier written in Python 2.7.5 and using the latest versions of Pandas and Numpy. To do a classification, run rfclassifier(), which takes as arguments a Pandas DataFrame for training the forest (sample), a test DataFrame to be classified (testsample), a list of attributes to use in building the tree (attributes), the name of your target column (target), and then optional arguments for the number of trees in the forest (N=1000), what percent of your training data to use as a validation set for pruning (alpha=0.30), a Boolean to turn off pruning (prune=30), and lastly the number of samples in each bootstrap used for a decision tree (n=len(sample)).

Using splitframe() on the classic ML Iris data set at https://archive.ics.uci.edu/ml/datasets/Iris to pick 30 samples for test data and training on the rest, rfclassifier() classifies with 100% accuracy using a sufficiently large bootstrap sample size. Without preprocessing of data, currently scores 78.469% accuracy for Titanic survival data on Kaggle: http://www.kaggle.com/c/titanic-gettingStarted

Currently working for numerical and categorical variables in python. For numerical categories with upwards of 5 unique values, branches will be split based on a single value and whether each point is above or below that value. To build a tree, run dtrees.py and mlforests.py and then use treebuilder(), which returns a Tree object. For inputs, treebuilder takes a sample in the form of a Pandas DataFrame, a list of the columns you want to train on as strings, and a string of the name of target column. Run treescore(Tree,sample,target) to determine the accuracy of the tree on a target sample. Training accuracy will print when you run treebuilder, but can also be computed accordingly. Recently made robust to missing entry values in training and classification by use of probabilistic label assignment. Numerical values will remain NA but categorical values will be physically assigned in training.

To build a forest of trees, use forestbuilder() with the same arguments as treebuilder() and an optional argument for the number of bootstrapped trees, N (defaulted to 100). This returns a list of trees. Then use forestclassify(forest,testdata) to return a list of classifications for your data set. Alternatively, using forestiter() will build trees ten at a time, classify with them, and then delete them to save memory. It then directly outputs results. As such, you will not actually have the trees used in this forest.

Soon to come: 
  -a query-based user interface for running the random forest classifier

Summary:  
MLForests is a hard-coded program for creating random forests from a Pandas dataframe, written solely by Tristan Williams. It was created to be both an exercise to refine my Python skills and a more customizable decision tree interface for statistical classification. It will be continuously updated, with the end goal of being an error-robust classifying algorithm for use in data science. dtrees is an object class containing Tree and Node for building a decision tree, while mlforests contains the functions necessary to bootstrap a sample from a Pandas dataframe and construct a random forest classifier from it.


