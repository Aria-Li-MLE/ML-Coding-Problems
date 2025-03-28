import numpy as np

class LinearRegression:
    """Linear regression model trained using gradient descent.
    
    Attributes:
        data (np.ndarray): Training data (normalized), shape (n_samples, n_features)
        labels (np.ndarray): Target values, shape (n_samples, 1)
        n_feas (int): Number of features
        theta (np.ndarray): Model parameters, shape (n_features, 1)
        cost_history (list): Record of cost function values during training
    """
    
    def __init__(self, data, labels):
        """Initialize linear regression model.
        
        Args:
            data: Normalized training data (numpy array)
            labels: Target values (numpy array)
        """
        self.data = data
        self.labels = labels.reshape(-1, 1)  # Ensure proper shape
        self.n_feas = self.data.shape[1]
        self.theta = np.zeros((self.n_feas, 1))
        self.cost_history = []
    
    def train(self, alpha=0.01, num_iterations=500, verbose=False):
        """Train the model using gradient descent.
        
        Args:
            alpha: Learning rate
            num_iterations: Number of iterations for gradient descent
            verbose: Whether to print progress
        """
        self.gradient_descent(alpha, num_iterations, verbose)
    
    def gradient_descent(self, alpha, num_iterations, verbose):
        """Perform gradient descent optimization.
        
        Args:
            alpha: Learning rate
            num_iterations: Number of iterations
            verbose: Whether to print progress
        """
        for i in range(num_iterations):
            self.gradient_step(alpha)
            current_cost = self.compute_cost()
            self.cost_history.append(current_cost)
            
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {current_cost:.4f}")
    
    def gradient_step(self, alpha):
        """Perform a single gradient descent update.
        
        Args:
            alpha: Learning rate
        """
        num_examples = self.data.shape[0]
        predictions = self.hypothesis(self.data)
        errors = predictions - self.labels
        gradient = (1/num_examples) * np.dot(self.data.T, errors)
        self.theta -= alpha * gradient
    
    def hypothesis(self, data):
        """Compute predictions for given data.
        
        Args:
            data: Input data (numpy array)
            
        Returns:
            Predictions (numpy array)
        """
        return np.dot(data, self.theta)
    
    def compute_cost(self):
        """Compute the mean squared error cost.
        
        Returns:
            MSE cost (float)
        """
        predictions = self.hypothesis(self.data)
        errors = predictions - self.labels
        return np.mean(errors**2)
    
    def predict(self, X):
        """Make predictions on new data.
        
        Args:
            X: Input data (normalized), shape (n_samples, n_features)
            
        Returns:
            Predictions (numpy array)
        """
        return self.hypothesis(X)