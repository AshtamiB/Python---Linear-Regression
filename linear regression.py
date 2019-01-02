# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:31:35 2018

@author: Ashtami
"""
import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as nppp
import scipy.stats as sps
from pylab import plot,show

DataFrame = pd.read_csv('C:/Users/Ashtami/Documents/Python/Insurance.csv',header=None)
DataMatrix = DataFrame.as_matrix()
InputMatrix= np.array(DataMatrix[:,0])
OutMatrix = np.array(DataMatrix[:,1])
(gradient,intercept,rvalue,pvalue,stderr) = sps.linregress(InputMatrix,OutMatrix)
Regression_line = nppp.polyval(InputMatrix,[intercept,gradient])
print ("Gradient & Intercept", gradient, intercept)
plot(InputMatrix,OutMatrix, 'vr')
plot(InputMatrix,Regression_line ,'b.-')
show()


#2nd method
import sklearn.linear_model as sklm
reg = sklm.LinearRegression()
reg.fit (X, y) # Fit the regression line
reg.predict(X_dash) # Make predictions using any Data


#REGRESSION METRICS
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_true = [3.1, -0.5, 2.0, 7]
y_pred = [2.7, 0.0, 1.8, 8]
print(mean_absolute_error(y_true, y_pred))
print(np.sqrt(mean_squared_error(y_true, y_pred)))
print(r2_score(y_true, y_pred))


#EXAMPLE LINEAR LASSO RIDGE
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pylab as pl
boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)
#print(boston.data)
x = boston.data
y = boston.target
#================== Split Data into Train & Test ==================
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#============================= OLS =================================
# Create linear regression object
OLS = LinearRegression()
# Train the model using the training sets
OLS.fit(X_train, y_train)
p_OLS_train = OLS.predict(X_train)
print('Regression Coefficients: \n', OLS.coef_)
MAE_OLS_Train = mean_absolute_error(y_train, p_OLS_train)
RMSE_OLS_Train= np.sqrt(mean_squared_error(y_train, p_OLS_train))
R2_OLS_Train = r2_score(y_train, p_OLS_train)
p_OLS_test = OLS.predict(X_test)
MAE_OLS_Test = mean_absolute_error(y_test, p_OLS_test)
RMSE_OLS_Test= np.sqrt(mean_squared_error(y_test, p_OLS_test))
R2_OLS_Test = r2_score(y_test, p_OLS_test)
#============================= Ridge ===============================
# Create linear regression object with a ridge coefficient 0.5
ridge = Ridge(fit_intercept=True, alpha=0.5) #Train the model using the training set
ridge.fit(X_train, y_train)
p_ridge_train = ridge.predict(X_train)
MAE_Ridge_Train = mean_absolute_error(y_train, p_ridge_train)
RMSE_Ridge_Train= np.sqrt(mean_squared_error(y_train,
p_ridge_train))
R2_Ridge_Train = r2_score(y_train, p_ridge_train)
p_ridge_test = ridge.predict(X_test)
MAE_Ridge_Test = mean_absolute_error(y_test, p_ridge_test)
RMSE_Ridge_Test= np.sqrt(mean_squared_error(y_test, p_ridge_test))
R2_Ridge_Test = r2_score(y_test, p_ridge_test)
#============================= Lasso ===============================
# Create linear regression object with a lasso coefficient 0.5
lasso = Lasso(fit_intercept=True, alpha=0.1) #Train the model using the training set
lasso.fit(X_train, y_train)
p_lasso_train = lasso.predict(X_train)
MAE_Lasso_Train = mean_absolute_error(y_train, p_lasso_train)
RMSE_Lasso_Train= np.sqrt(mean_squared_error(y_train, 
p_lasso_train))
R2_Lasso_Train = r2_score(y_train, p_lasso_train)
p_lasso_test = lasso.predict(X_test)
MAE_Lasso_Test = mean_absolute_error(y_test, p_lasso_test)
RMSE_Lasso_Test= np.sqrt(mean_squared_error(y_test, p_lasso_test))
R2_Lasso_Test = r2_score(y_test, p_lasso_test)
#====================== Print Error Criteria =======================
print('-----------------OLS------------------')
print(MAE_OLS_Test)
print(RMSE_OLS_Test)
print(R2_OLS_Test)
print('----------------Ridge-----------------')
print(MAE_Ridge_Test)
print(RMSE_Ridge_Test)
print(R2_Ridge_Test)
print('----------------Lasso-----------------')
print(MAE_Lasso_Test)
print(RMSE_Lasso_Test)
print(R2_Lasso_Test)
#========================= Plot Outputs ============================
#matplotlib inline
pl.plot(p_OLS_train , y_train,'ro')
pl.plot(p_ridge_train , y_train,'bo')
pl.plot(p_lasso_train , y_train,'co')
pl.plot([0,50],[0,50] , 'g-')
pl.xlabel('Predicted')
pl.ylabel('Real Target Values')
pl.show()