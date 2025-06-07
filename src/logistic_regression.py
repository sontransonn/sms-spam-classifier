import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0
        self.costs = [] 

    def sigmoid(self, z): 
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred):
        epsilon = 1e-10
        cost = - (1 / len(y_true)) * np.sum(
            y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.costs = [] 
        
        for i in range(self.num_iter):
            z = np.dot(X, self.weights) + self.bias 
            y_pred = self.sigmoid(z) 

            cost = self._compute_cost(y, y_pred)
            self.costs.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 100 == 0:
                print(f"Epoch {i + 1}/{self.num_iter}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        prob_class1 = self.sigmoid(z)
        prob_class0 = 1 - prob_class1
        return np.column_stack((prob_class0, prob_class1))

    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z) 
        return (y_pred >= threshold).astype(int)