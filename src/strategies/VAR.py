from scipy.optimize import least_squares
import numpy as np
from scipy.linalg import circulant



    
class trainingdata: #feed dataset as [timeaxis, features]
    def __init__(self, data: np.ndarray, max_lag: int = 10):
        self.data = np.roll(circulant(data.T),-1,axis=2)[:,:,-max_lag:][:,:-max_lag-2,::-1]
        self.validation = data[max_lag+2:]

    def __getitem__(self, idx):
        return self.data[idx]
    
    def predictor_sweep(self, prediction_matrix: np.ndarray, offset: np.ndarray):
        return np.sum(self.data[:,:,-prediction_matrix.shape[0]:] @ prediction_matrix, axis=2).T + offset

class VARmodel:
    def __init__(self, num_vars, max_lag=10):
        self.num_vars = num_vars
        self.max_lag = max_lag
        self.predictors = [] 
        self.log_Sigma = [0,]*max_lag
        for lag in range(1, max_lag + 1):
            self.predictors.append(np.ones((lag+1,num_vars)))  # Initialize with ones
    
    def load_training_data(self, data: np.ndarray):
        self.training_data = trainingdata(data, max_lag=self.max_lag)

    def fit(self):
        # Fit the VAR model to the data
        for lag in range(1, self.max_lag + 1):
            residuals = lambda predicting_matrix: (self.training_data.predictor_sweep(np.reshape(predicting_matrix[:lag* self.num_vars],(lag,self.num_vars)), np.reshape(predicting_matrix[-self.num_vars:],(1,self.num_vars))) - self.training_data.validation).flatten()
            result = least_squares(residuals, self.predictors[lag-1].flatten())
            self.predictors[lag-1] = np.reshape(result.x,(lag+1, self.num_vars))
            sigma = result.residuals.reshape(-1, self.num_vars).T @ result.residuals.reshape(-1, self.num_vars) / len(results.residuals// self.num_vars)
            sign, logdet = np.linalg.slogdet(sigma)
            self.log_Sigma[lag-1] = logdet

    def optimal_lag(self):
        # Use BIC to determine the optimal lag
        logS = np.array(self.log_Sigma)
        T = self.training_data.validation.shape[0]
        BIC = logS + np.log(T) * (np.arange(1, self.max_lag + 1) * self.num_vars**2) / T
        print("optimal lag:", np.argmin(BIC)+1)





