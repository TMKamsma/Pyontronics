import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

class LinearAutoregression:
    def __init__(self, window_size=10, lookback_stride=1, prediction_horizon=10):
        """
        AR model for ventilator pressure tasks
        
        Parameters:
        window_size: Number of past samples to use as features
        lookback_stride: Use 1 in N past samples (e.g., 5 for sparse lookback)
        prediction_horizon: Steps ahead to predict (in samples)
        """
        self.window_size = window_size
        self.lookback_stride = lookback_stride
        self.prediction_horizon = prediction_horizon
        self.model = LinearRegression()
        
    def _create_features(self, data):
        """Create AR features with optional sparse lookback"""
        n_samples = len(data) - self.window_size * self.lookback_stride - self.prediction_horizon + 1
        X = np.zeros((n_samples, self.window_size))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Create window with stride (sparse lookback)
            window_indices = range(i, i + self.window_size * self.lookback_stride, self.lookback_stride)
            X[i] = data[window_indices]
            y[i] = data[i + self.window_size * self.lookback_stride + self.prediction_horizon - 1]
            
        return X, y
    
    def _create_classification_features(self, pressure_data, valve_states):
        """Create features for valve state classification"""
        n_samples = len(pressure_data) - self.window_size * self.lookback_stride + 1
        X = np.zeros((n_samples, self.window_size))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            window_indices = range(i, i + self.window_size * self.lookback_stride, self.lookback_stride)
            X[i] = pressure_data[window_indices]
            y[i] = valve_states[i + self.window_size * self.lookback_stride - 1]
            
        return X, y
    
    def train_pressure_prediction(self, pressure_data):
        """Train model for pressure prediction"""
        X, y = self._create_features(pressure_data)
        self.model.fit(X, y)
        
    def train_valve_classification(self, pressure_data, valve_states):
        """Train model for valve state classification"""
        X, y = self._create_classification_features(pressure_data, valve_states)
        self.model.fit(X, y)
        
    def predict_pressure(self, pressure_data):
        """Predict pressure values"""
        X, y = self._create_features(pressure_data)
        return self.model.predict(X)
    
    def predict_valve_state(self, pressure_data, threshold=0.5):
        """Predict valve states (0/1)"""
        X, _ = self._create_classification_features(pressure_data, np.zeros(len(pressure_data)))
        return (self.model.predict(X) > threshold).astype(int)
    
    def evaluate_pressure(self, pressure_data):
        """Evaluate pressure prediction performance"""
        X, y = self._create_features(pressure_data)
        pred = self.model.predict(X)
        return mean_squared_error(y, pred), np.corrcoef(y, pred)[0,1]
    
    def evaluate_valve(self, pressure_data, valve_states):
        """Evaluate valve classification performance"""
        X, y = self._create_classification_features(pressure_data, valve_states)
        pred = (self.model.predict(X) > 0.5).astype(int)
        return accuracy_score(y, pred)
