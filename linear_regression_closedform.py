from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

##Load dataset
X,y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

##closed form solution
class LinearRegression:
    def __init__(self):
        self.coef=None
        self.intercept=None
    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        output=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.coef=output[1:]
        self.intercept=output[0]
    def predict(self,X_test):
        return np.dot(X_test,self.coef)+self.intercept
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

r2_score=r2_score(y_test,y_pred)
print(r2_score)
