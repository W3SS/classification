from Classifier import *
import numpy as np
import math

class LogisticRegressionClassifier(Classifier):

    def learn(self, X, y):
        """
        Performs an iterative gradient ascent on the weight values.
        
        Args:
	       X: A list of feature arrays where each feature array corresponds to feature values
		of one observation in the training set.
	       y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
		to a negative instance of the class at said 0 or 1s index in featuresList.

	Returns: Nothing
        """
        numTrainingPairs = len(y)
        inputVars = len(X[0])
        self.w = np.empty(inputVars)
        self.w.fill(.0001)
         # learning rate
        self.eta = .001
        # convergence threshold
        self.epsilon = 0.01
        w_change = 1
        w_change_avg_last = .0000001

        while w_change > self.epsilon:

            last_w = np.copy(self.w)
            z_vals = np.empty(numTrainingPairs) #array storing summation terms
            z_vals.fill(0)
            #store summation terms first 
            for z in range(0,numTrainingPairs):
                z_vals[z] = np.dot(self.w, X[z])
            
            gradient = np.empty(inputVars)
            gradient.fill(0)
            #compute the batch gradient vector for each gradient
            for k in range(0, inputVars):
                #for each training instance
                for i in range(0, numTrainingPairs):
                    p = 1 / (1+math.exp(-1*z_vals[i]))
                    gradient[k] += X[i][k]*(y[i]-p)


            for b in range(0, inputVars):
                self.w[b] += self.eta * gradient[b]

            
            
            w_change_av = np.sum(abs(np.subtract(last_w, self.w)))/float(inputVars)
            
            

            #look at the the percentage difference between this 
            #iteration's average weight change and the previous 
            #iteration's average weight change

            w_change = abs(w_change_av-w_change_avg_last)/w_change_avg_last
            
            w_change_avg_last = w_change_av



             
                                    
            




        # YOU IMPLEMENT

    def getLogProbClassAndLogProbNotClass(self, x):
        """
        Args:
            features: A numpy array that corresponds to the feature values for a single observation.

        Returns:
            A tuple containing the log probability that the observation is a member of the class
                and the log probability that the observation is NOT a member of the class
        """

         # YOU IMPLEMENT
        logProbClass = math.log(.5)
        logProbNotClass = math.log(.5)

        z = np.dot(self.w , x)
        logProbClass = 1/(1+ math.exp(-1*z))
        logProbNotClass = 1-logProbClass


        return (logProbClass, logProbNotClass)
