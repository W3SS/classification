from Classifier import *
import math
import random
import numpy as np

class DecisionTreeClassifier(Classifier): 

    def learn(self, X, y):
        """
        Constructs a decision tree.

        Args:
           X: A list of feature arrays where each feature array corresponds to feature values
        of one observation in the training set.
           y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
        to a negative instance of the class at said 0 or 1s index in featuresList.
        """
        DT = TreeNode(X, y, 0)
        DT.makeTree()
        self.DT = DT

    def getLogProbClassAndLogProbNotClass(self, x):
        """Returns log probabilities that a given observation is a positive sample or negative sample"""
        return self.DT.getLogProbClassAndLogProbNotClass(x)

class TreeNode: 

    def __init__(self, X, y, depth):
        self.X = X  # set of featurized observations
        self.y = y  # set of labels associated with the observations 
        self.depth = depth
        self.depthLimit =10  # limits the depth of your tree for the sake of performance; feel free to adjust
        self.n = len(X)
        self.splitFeature, self.children = None, None  # these attributes should be assigned in splitNode()
        self.entropySplitThreshold = 0.7219 # node splitting threshold for 80%/20% split; feel free to adjust

    def splitNode(self, splitFeature):
        ''' Creates child nodes, splitting the featurized data in the current node on splitFeature. 
        Must set self.splitFeature and self.children to the appropriate values.

        Args: splitFeature, the feature on which this node should split on (this should be the feature you obtain from
            the bestFeature() function)
        Returns: returns True if split is performed, False if not.
        '''
        if len(set(self.y)) < 2: # fewer than 2 labels in this node, so no split is performed (node is a leaf)
            return False
        
        #put negative features in left child positive in right
        left_child_Xs = []
        left_y = []
        right_child_Xs = []
        right_y = []

        
        for i in range(0, len(self.y)):
            feature_arr = self.X[i]
            if feature_arr[splitFeature] == 0:
                left_y.append(self.y[i])
                left_child_Xs.append(feature_arr)
            else:
                right_y.append(self.y[i])
                right_child_Xs.append(feature_arr)


        
        leftTree = TreeNode(left_child_Xs, left_y, self.depth+1)
        rightTree = TreeNode(right_child_Xs, right_y, self.depth+1)

        

        self.splitFeature = splitFeature
        self.children = (leftTree, rightTree)

        return True



    def bestFeature(self):
        ''' Identifies and returns the feature that maximizes the information gain.
        You should calculate entropy values for each feature, and then return the feature with highest entropy.
        Consider thresholding on an entropy value -- that is, select a target entropy value, and if no feature 
        has entropy above that value, return None as the bestFeature 

        Returns: the index of the best feature based on entropy
        '''
        num_data_points = len(self.X)

        if num_data_points == 0:
            return None

        
        num_vars = len(self.X[0])

        pos_counts = np.zeros(num_vars)
        neg_counts = np.zeros(num_vars)
        entropies = np.zeros(num_vars)

        for c in range(0, len(self.X[0])):
            neg_count = 0
            pos_count = 0
            total = 0
            for r in range(0, len(self.X)):
                total += 1
                if(self.X[r][c] == 0):
                    neg_count += 1
                else:
                    pos_count += 1
            pos_prob = float(pos_count)/float(len(self.X))
            neg_prob =  float(neg_count)/float(len(self.X))


            if(pos_prob > 0 and pos_prob < 1):
                entropies[c] = (pos_prob * -math.log(pos_prob, 2))+(neg_prob*-math.log(neg_prob,2))





        max_e = np.nanmax(entropies)
        

        if max_e > self.entropySplitThreshold:
            return np.argmax(entropies)

        else:
            return None


    def makeTree(self):
        '''Splits the root node on the best feature (if applicable),
        then recursively calls makeTree() on the children of the root.
        If there is no best feature, you should not perform a split, and this
        node will become a leaf'''



        bestFeature = self.bestFeature()

        if self.depth <= self.depthLimit and not bestFeature is None:
            if not self is None:
                was_split = self.splitNode(bestFeature)
                if(was_split and (not self.children == None)):
                    self.children[0].makeTree()
                    self.children[1].makeTree()

   
    

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            x: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """

        # YOU IMPLEMENT
        treeNode = self
        oldTree = self
        while(not treeNode.children == None):
            if x[treeNode.splitFeature] > 0:
                #traverse down the right side for positive
                treeNode = treeNode.children[1]
            else:
                #traverse down the left side for negative
                treeNode = treeNode.children[0]

        #once you are done traversing DT represents a leaf, so calcualte probs

        num_y1 = 0
        total = 0
        for y_i in treeNode.y:
            total += 1
            if y_i == 1:
                num_y1 += 1

        
        probClass = float(num_y1)/float(total)
        probNotClass = 1-probClass
        return (probClass, probNotClass)
