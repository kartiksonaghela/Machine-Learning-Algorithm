from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the data
X, y = load_diabetes(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression class
class LinearRegression:
    def __init__(self, epochs=1000, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    
    def fit(self, X_train, y_train):
        self.coef = np.ones(X_train.shape[1])
        self.intercept = 0
        
        # Gradient descent loop
        for _ in range(self.epochs):
            y_pred = np.dot(X_train, self.coef) + self.intercept
            
            # Compute gradients
            intercept_derivative = -2 * np.mean(y_train - y_pred)
            coeff_derivative = -2 * np.dot((y_train - y_pred), X_train) / X_train.shape[0]
            
            # Update weights
            self.intercept = self.intercept - self.learning_rate * intercept_derivative
            self.coef = self.coef - self.learning_rate * coeff_derivative
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef) + self.intercept

# Instantiate and train the model
model = LinearRegression(learning_rate=0.5)
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict(X_test)  # Fix: use X_test here, not y_test

# Evaluate the model
print(r2_score(y_test, prediction))
