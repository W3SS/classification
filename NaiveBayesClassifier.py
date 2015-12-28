from Classifier import *
import numpy as np
import math

"""
Your NBClassifier dude...
"""
class NaiveBayesClassifier(Classifier):

	def learn(self, X, y):

		"""
		You should set up your various counts to be used in classification here: as detailed in the handout.
		Args: 
			X: A list of feature arrays where each feature array corresponds to feature values
				of one observation in the training set.
			y: A list of ints where 1s correspond to a positive instance of the class and 0s correspond
				to a negative instance of the class at said 0 or 1s index in featuresList.

		Returns: Nothing
		"""

		 # YOU IMPLEMENT -- CORRECTLY GET THE COUNTS
		self.occurencesOfClass = 0

		self.occurencesOfNotClass = 0

		self.totalFeatureCountsForClass = np.zeros(len(X[0]), np.int32)
		self.totalFeatureCountsForNotClass = np.zeros(len(X[0]), np.int32)

		self.totalOccurencesOfFeatureInClass = 0
		self.totalOccurencesOfFeatureInNotClass = 0


       
		index = 0
		leny = len(y)
		lenx = len(X)
        

		for feature_arr in X:

			if y[index] == 0:
				self.occurencesOfNotClass += 1
				for i in range(0, len(feature_arr)):
					self.totalFeatureCountsForNotClass[i] += feature_arr[i]
					self.totalOccurencesOfFeatureInNotClass += feature_arr[i]
                    
			else:
				self.occurencesOfClass += 1
				for i in range(0, len(feature_arr)):
					self.totalFeatureCountsForClass[i] += feature_arr[i]
					self.totalOccurencesOfFeatureInClass += feature_arr[i]

			index += 1
			

		self.totalFeatureObservations = self.totalOccurencesOfFeatureInClass + self.totalOccurencesOfFeatureInNotClass

#current error: getLogProbClassAndLogProbNotClass logProbClass += math.log(p_y) ValueError: math domain error

	def getLogProbClassAndLogProbNotClass(self, x):
		"""
		You should calculate the log probability of the class/ of not the class using the counts determined
		in learn as detailed in the handout. Don't forget to use epsilon to smooth when a feature in the 
		observation only occurs in only the class or only not the class in the training set! 

		Args: 
			x: a numpy array corresponding to a featurization of a single observation 
			
		Returns: A tuple of (the log probability that the features arg corresponds to a positive 
			instance of the class, and the log probability that the features arg does not correspond
			to a positive instance of the class).
		"""		
		# YOU IMPLEMENT -- CORRECTLY GET THE COUNTS
		epsilon = 1/ float(self.totalFeatureObservations)
		logProbClass = 0
		logProbNotClass = 0
		p_y = self.occurencesOfClass/float(self.occurencesOfClass + self.occurencesOfNotClass)
		logProbClass += math.log(p_y)
		logProbNotClass += math.log(1-p_y)

		index = 0
		for x_i in x:
			num_features_in_class = self.totalFeatureCountsForClass[index]

			#p(x_i | y) = #occurrences of feature x_i in class y/ total # occurs of features in class y
			p_x_given_y = float(num_features_in_class) / self.totalOccurencesOfFeatureInClass
			#add the log prob of p_x_given_y raised to the power of the num times it is this feature x
			if num_features_in_class == 0:
				logProbClass += math.log(epsilon)*x_i
			else:
				logProbClass += math.log(p_x_given_y)*x_i
			#do the same thing for not
			num_features_in_not_class = self.totalFeatureCountsForNotClass[index]
			p_x_given_not_y = float(num_features_in_not_class) / self.totalOccurencesOfFeatureInNotClass

			if num_features_in_not_class == 0:
				logProbNotClass += math.log(epsilon)*x_i
			else:
				logProbNotClass += math.log(p_x_given_not_y)*x_i
            
			index += 1             


		return (logProbClass, logProbNotClass)







