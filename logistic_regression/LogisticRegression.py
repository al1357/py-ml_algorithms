import numpy as np
import matplotlib.pyplot as plt
from numpy import float64
"""
To do:
 - implement regularization
 - implement testing
 
Questions:
 - is it ok to normalize prediction features like that? They're normalized using training mean, stdDev and m.
   Or should I compute new numbers from set of trainig data + prediction data?
   
Matlab results(coursera) - see bottom.
"""
class LogisticRegression:
    # x (n, m)
    x = None
    # labels (1, m) for l=1
    y = None
    # bias scalar
    b = None
    # thetas (1, n)
    theta = None
    # learing rate
    alpha = None
    # number of rows/training examples
    m = None
    # number of features without bias
    n = None
    
    def __init__(self, data_train, labels_train, data_test):
        # Select all rows and columns from 0 to the last x index - if the rest is y
        # T transpose to (n, m)
        self.x = data_train.T
        shape = np.shape(self.x) # shape (n, m)
        self.m = shape[1]
        self.n = shape[0]
        # T to (l, m)
        self.y = labels_train.T
        self.theta = np.zeros((1, self.n), dtype="float64")
        self.b = 1
        self.alpha = 0.1
        
        self.x_test = data_test.T
    #end
    
    def sigmoid(self, z):
        """Sigmoid function
        prediciton: sigmoid >= 0.5 result 1; sigmoid < 0.5 result 0;
        """
        return 1 / (1 + np.exp(-z))
    #end
    
    def featuresNormalize(self, x, prediction=False): 
        """Normalizes features: >>> (feature - mean) / standard deviation <<<
        """
        if prediction == False:
            self.xMean = np.sum(self.x,1).reshape((self.n, 1)) / self.m         # (n, 1)
            self.stdDev = np.std(self.x,1).reshape((self.n, 1))                 # (n, 1)
        xNormal = (x - self.xMean) / self.stdDev;                               # ((n, m) - (n, 1)) / (n, 1)
        return xNormal
    #end
    
    def computeCost(self, x, y):
        """Computes cost function
            x (n, m)
        """
        m = x.shape[1]
        z = np.dot(self.theta, x) + self.b # (1, n)@(n, m) = (1, m)
        a = self.sigmoid(z) # (1, m) > SIGMOID - converts prediction to 0-1 range
        # log(1) = 0 - the lowest possible outcome; 
        # for y=1 log(a) is close to 0 when a -> 1; 
        # for y=0 log(1-a) is colose to 0 when a -> 0;
        # in log(a) if a < 1 then log(a) < 0 - hence we multiply by -y or -(1-y), to make the result positive;
        
        # prediction error for labels y=1 - we multiply other results by y=0;
        # the further sigmoidResult is from 1 -> 0, the bigger(negative) log(sigmoidResult)
        predictionForOnes =  np.dot(-y, np.log(a).T) # (1, m) x (m, 1)
        # prediction error for labels y=0 - we multiply other results by (1-y) = (1-1) = 0
        # the further sigmoidResult is from 0 the bigger(negative) log(1-sigmoidResult)
        
        
        # np.log of very small number results in division by 0 error
        # https://stackoverflow.com/questions/46510929/mle-log-likelihood-for-logistic-regression-gives-divide-by-zero-error
        
        predictionForZeroes = np.dot((1. - y), np.log(1. - a).T)
        cost = ( predictionForOnes - predictionForZeroes) / m    # (1, 1) > COST; convert to scalar with .max()
        return cost.max()
    #end
    
    def computeDz(self):      
        """Compute partial derivative dz on the training set - how J changes with respect to W(?)
        """
        a = self.predict(self.x)
        # a prediction 0-1; self.y 0 or 1 label
        return a - self.y
    #end
    
    def computeGradient(self, dz):
        """Computes gradient on the training set
        """
        gradient = (np.dot(self.x, dz.T)) / self.m # (n, m)@(1, m).T = (n, 1)
        return gradient
    #end
  
    def runTraining(self):
        """Runs gradient descent by iterating over gradient function.
        """
        print("Initial cost: ",self.computeCost(self.x, self.y))
        dz = self.computeDz()
        print("Initial bias gradient: ",(np.sum(dz).max() / self.m))
        print("Initial theta gradient: ",self.computeGradient(dz))
        #optX = np.asarray(self.x)
      
        for i in range(1500):
            dz = self.computeDz()
            self.theta = self.theta - (self.alpha * self.computeGradient(dz))
            self.b = self.b - (self.alpha * (np.sum(dz).max() / self.m))
            
        print("Final cost: ",self.computeCost(self.x, self.y))
        print("Final bias: ",self.b)
        print("Final weights: ",self.theta)
    #end
    
    def report(self):
        print('a')
    #end
    
    def predict(self, x):
        p = self.sigmoid(np.dot(self.theta, x) + self.b)
        return p
    #end
    
    def getAccuracy(self, x, y):
        p = self.predict(x)
        pBinary = np.where(p>=0.5,1,0)
        pyCompare = (pBinary == y)
        accuracy = np.mean(pyCompare) * 100
        print("Accuracy: ",accuracy)  
    #end
    
    def predictTest(self):
        student = np.array([[45], [85]], dtype="float64")
        studentNorm = self.featuresNormalize(student, True)
        print("Normalized student scores: ",studentNorm)
        studentPrediction = self.predict(studentNorm)
        print("Probability of 1: ",studentPrediction)
    #end
    
    """
    Matlab results(coursera) - without regularization and feature scaling:
        Cost at initial theta (zeros): 0.693147
        Gradient at initial theta (zeros): 
         -0.100000 
         -12.009217 
         -11.262842 
         
         Cost at theta found by fminunc: 0.203506
        theta: 
         -24.932774 
         0.204406 
         0.199616 
    
        For a student with scores 45 and 85, we predict an admission probability of 0.774321

        Train Accuracy: 89.000000
        ('Train Accuracy: %f\n', mean(double(p == y)) * 100) - double() converts logical into numbers
    """
        