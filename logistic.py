from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset and split into train and test sets
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr  # Learning rate
        self.num_iter = num_iter  # Number of iterations
        self.weights = None  # Will store weights and intercept

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid function

    def fit(self, X_train, y_train):
        # Insert bias term (column of 1s) at the start
        X = np.insert(X_train, 0, 1, axis=1)
        n_samples, n_features = X.shape
        
        # Initialize weights (including intercept as part of the weight vector)
        self.weights = np.ones(n_features)
        
        # Gradient descent optimization
        for _ in range(self.num_iter):
            y_hat = self.sigmoid(np.dot(X, self.weights))  # Predicted values
            
            # Gradient of the loss function
            gradient = np.dot(X, (y_hat - y_train)) / n_samples
            
            # Update weights
            self.weights -= self.lr * gradient

    def predict(self, X_test):
        # Insert bias term (column of 1s) at the start
        X = np.insert(X_test, 0, 1, axis=1)
        y_pred = self.sigmoid(np.dot(X, self.weights))  # Raw predictions
        
        # Convert probabilities to 0 or 1
        return np.where(y_pred >= 0.5, 1, 0)

# Instantiate and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")