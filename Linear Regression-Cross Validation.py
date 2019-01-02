# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:41:31 2018

@author: Ashtami
"""
import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_boston
import numpy as np
boston = load_boston()
x = boston.data
y = boston.target
#========================= OLS =============================
print('-----------------------------------------------')
linreg = LinearRegression()
linreg.fit(x, y)
p = linreg.predict(x)
# Compute RMSE using 10-fold cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train, test in kf:
 linreg.fit(x[train], y[train])
 p = linreg.predict(x[test])
 e = p - y[test]
 xval_err += np.dot(e, e)
rmse_10cv = np.sqrt(xval_err / len(x))
print('OLS RMSE on 10-fold CV: %.4f' %rmse_10cv)
#======================= Ridge =============================
print('-----------------------------------------------')
ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(x, y)

#Exercise – Week VI
#Data Programming With Python – Fall / 2018
p = ridge.predict(x)
# Compute RMSE using 10-fold Cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
 ridge.fit(x[train],y[train])
 p = ridge.predict(x[test])
 e = p-y[test]
 xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))
print('Ridge RMSE on 10-fold CV: %.4f' %rmse_10cv)
#======================== Lasso ============================
print('-----------------------------------------------')
lasso = Lasso(fit_intercept=True, alpha=0.1)
lasso.fit(x, y)
p = lasso.predict(x)
# Compute RMSE using 10-fold cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
 lasso.fit(x[train],y[train])
 p = ridge.predict(x[test])
 e = p-y[test]
 xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))
print('Lasso RMSE on 10-fold CV: %.4f' %rmse_10cv)
