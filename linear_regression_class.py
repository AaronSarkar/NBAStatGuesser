# This is a linear regression class I created to understand how the algorithm works
import numpy as np
import math
class linear_regression:
    def __init__(self):
        self.weights = []
        self.bias = np.random.rand() - 0.5
    
    def __str__(self):
        return f"weights: {self.weights}, bias: {self.bias}"
    
    def calculate_single(self, x_train):
        try:
            x = x_train.values.reshape((-1,1))
            function = np.dot(x, self.weights)
            function += self.bias
            return function
        except:
            print("Invalid or incompatable inputs")
            return None
        
    def r2_score(self, x_train, y_train):
        try:
            x_train = x_train.values
            y_train = y_train.values
            
            y_pred = np.matmul(x_train, self.weights)
            [x+self.bias for x in y_pred]
            y_train = y_train.reshape((-1,1))
            
            numerator = np.sum((y_train-y_pred)**2)
            y_mean = np.mean(y_pred)
            denominator = np.sum((y_train-y_mean)**2)

            return 1 - numerator/denominator
        
        except:
            print("Invalid or incompatable inputs")
            return None

    def normalize(self, x_train):
        try:
            mean = np.mean(x_train.values, axis = 0)
            deviation = (np.std(x_train.values, axis = 0))
            normalization = (x_train.values - mean)/deviation
            x_train.iloc[:, :] = normalization
            return x_train
        except:
            print("Invalid input")
            return None
    
    def batch_backpropagation(self, x_train, y_train, iterations, alpha):
        try:
            x_train = x_train.values
            y_train = y_train.values
            y_train = y_train.reshape((-1,1))
            length = np.size(x_train, axis = 1)
            self.weights = np.random.rand(length,1) - 0.5
            m = len(x_train)
            
            for i in range(iterations):
                function = np.matmul(x_train, self.weights)
                function += self.bias
                function = function - y_train
                dcost_df = function/(m)
                dj_db = np.sum(dcost_df)
                dj_dw = np.dot(np.transpose(x_train),  dcost_df)
                self.bias = self.bias - alpha*dj_db
                self.weights = self.weights - alpha*dj_dw
        except:
            print("Invalid or incompatable inputs")
    

    

    

    