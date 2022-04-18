# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:56:08 2021

@author: acarr
"""
import numpy as np
import pandas as pd
from illumina_combiner import ilmn_final
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


Y = np.array(ilmn_final['Open'])
X = np.array(ilmn_final.drop(['Open'], 1))


scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
X = scalerX.fit_transform(X)
Y = Y.reshape(-1, 1)
#Y = scalerY.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_pred = X_test[0:20]
Y_check = Y_test[0:20]


#Performing the Regression on the training data
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, Y_train)
prediction = (lin_regressor.predict(X_pred))
Y_pred = lin_regressor.predict(X_test)
#print(prediction)
#print(scalerY.inverse_transform(prediction))
#print('Prediction Score: ', lin_regressor.score(X_test, Y_test))
#print('Mean Squared Error : ', mean_squared_error(Y_test, Y_pred))
diff = 0
for i in range(0, len(Y_check)):
    diff = diff+abs(prediction[i]-Y_check[i])
percent_error = diff/20
print(percent_error)

for i in range(0, len(Y_test)):
    diff = diff+abs(Y_pred[i]-Y_test[i])
percent_error = diff/20
print(percent_error)