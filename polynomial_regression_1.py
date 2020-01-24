# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:08:34 2020

@author: Dolley
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#linear regression taining set
from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X,y)

#testing set
#y_pred=linearRegression.predict(X_test)

#visualisation of the data in hte linear regression format for the training set 
plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,linearRegression.predict(X_train),color='green')
plt.title('Truth or Bluff (Linear Regression from training set)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualisation of the data that is in form of linear regression
plt.scatter(X,y,color='black')
plt.plot(X,linearRegression.predict(X),color='green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)#transforming the x values
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Predicting a new result with Linear Regression
#error occured couldnt get the predicted value yet to slv
linearRegression.predict([[6.5]])

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))