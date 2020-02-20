import numpy as np
import matplotlib.pyplot as plt

class linear_regression:
    X = None
    # raw x
    X_raw = None
    # normalized x
    X_normal = None
    
    Y = None
    # number of rows
    m = None
    # number of features without bias
    n = None
    x_mean = None
    std_dev = None
    cost ={ 'mae': np.array([]), 'mse': np.array([]), 'rmse': np.array([]), 'mape': np.array([]),'mpe': np.array([]) }
    theta = None
    iterations = 1500
    alpha = 0.01
    
    def __init__(self, train_samples, Y, normalize=False, alpha=0.01, iterations=1500):
        self.alpha = alpha
        self.iterations = iterations
        self.X_raw = train_samples
        self.Y = Y
        self.m = self.X_raw.shape[0]
        bias = np.ones((self.m, 1));
        self.n = self.X_raw.shape[1]
        self.theta = np.zeros((self.n+1, 1))
        if normalize:
            self.features_normalize()
            self.X = np.concatenate((bias, self.X_normal), 1)
        else:
            self.X = np.concatenate((bias, self.X_raw), 1)
    #end
    
    def plot(self, x_label, y_label):
        """Plot 2D input/output data
        """
        plt.scatter(self.X_raw, self.Y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    #end
    
    def error(self, kind, X=None, Y=None, showDetails=False):
        X = self.X if (X is None) else X
        Y = self.Y if (Y is None) else Y
        if(showDetails):
            print(self.predict(X))
            print(Y)
            print((self.predict(X) - Y))
        if(kind == 'mae'):
            return self.mean_absolute_error(X, Y)
        elif(kind == 'mse'):
            return self.mean_square_error(X, Y)
        elif(kind == 'rmse'):
            return self.root_mean_square_error(X, Y)
        elif(kind == 'mape'):
            return self.mean_absolute_percentage_error(X, Y)
        elif(kind == 'mpe'):
            return self.mean_percentage_error(X, Y)
        #endif
    #end error
    
    def mean_absolute_error(self, X, Y):
        diff = (self.predict(X) - Y)
        return np.sum(diff) / (2 * self.m)
    #end
    
    def mean_square_error(self, X, Y):
        diffPow = np.power((self.predict(X) - Y), 2)
        return np.sum(diffPow) / (2 * self.m)
    #end
    
    def root_mean_square_error(self, X, Y):
        return np.sqrt(self.mean_square_error(X, Y))
    #end
    
    def mean_absolute_percentage_error(self, X, Y):
        pred = self.predict(X)
        pred[pred==0.] = np.finfo(np.float64).min
        return np.sum(np.abs((pred-Y)/pred)) / self.m
    #end
    
    def mean_percentage_error(self, X, Y):
        pred = self.predict(X)
        pred[pred==0.] = np.finfo(np.float64).min
        return np.sum((pred-Y)/pred) / self.m
    #end
    
    def plot_cost(self):
        """Plot cost agains number of iterations
        """
        plt.plot(self.cost['mse'])
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
        
        plt.plot(self.cost['mae'], '-b', label="mae")
        plt.plot(self.cost['rmse'], '-g', label="rmse")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
        
        plt.plot(self.cost['mape'], '--r', label="mape")
        plt.plot(self.cost['mpe'], '--c', label="mpe")
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
    #end
    
    def features_normalize(self): 
        """Normalizes features: (feature - mean) / standard deviation
        """
        self.x_mean = np.sum(self.X_raw, 0) / self.m
        self.std_dev = np.std(self.X_raw, 0)        
        x_norm = []
        for i in range(self.m):
            normalized_row = self.row_normalize(self.X_raw[i])
            x_norm.append(normalized_row)
        self.X_normal = np.array(x_norm)    
        return self.X_normal
    #end
    
    def row_normalize(self, row=[]):
        """Normalizes single row of features provided as parameter
        """
        return ((row - self.x_mean) / self.std_dev)
    #end    
    
    def gradient_descent(self):
        """Run gradient descent
        """
        for i in range(self.iterations):
            temp_cost_mse = self.error('mse')
            if(i==0):
                print("Initial MSE: ", temp_cost_mse)
            self.cost['mse'] = np.append(self.cost['mse'], temp_cost_mse)
            self.cost['rmse'] = np.append(self.cost['rmse'], self.error('rmse'))
            self.cost['mae'] = np.append(self.cost['mae'], self.error('mae'))
            self.cost['mape'] = np.append(self.cost['mape'], self.error('mape'))
            self.cost['mpe'] = np.append(self.cost['mpe'], self.error('mpe'))
            temp_diff_out = self.X.dot(self.theta) - self.Y
            temp_diff_in = None
            
            for j in range(self.n+1):
                if j!=0:
                    # if we're not calculating thetas for bias term
                    temp_diff_in = temp_diff_out * self.X[:,[j]] 
                else:
                    temp_diff_in = temp_diff_out
                    
                self.theta[j] = self.theta[j] - ((self.alpha * np.sum(temp_diff_in)) / self.m)
            #end for    
                
        print("Final mse: ",self.cost['mse'][-1])
        print("Final rmse: ",self.cost['rmse'][-1])
        print("Final mae: ",self.cost['mae'][-1])
        print("Final theta: ", self.theta)
        self.plot_cost()
    #end
    
    def predict(self, data_in, normalize=False):
        if(normalize):
            data_in = self.row_normalize(data_in)
            data_in = np.insert(data_in, 0, 1, axis=1)
        #end if
        return data_in.dot(self.theta)
    #end    
    
    def print_data(self, limit=10):
        for i in range(0,limit):
            print("X: ",self.X_raw[i],";     y: ",self.Y[i])
    #end