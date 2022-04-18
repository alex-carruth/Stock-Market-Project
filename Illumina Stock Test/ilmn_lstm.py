# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:42:02 2021

@author: acarr
"""
import numpy as np
import pandas as pd
from illumina_combiner import ilmn_final
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

Y = np.array(ilmn_final['Open'])
X = np.array(ilmn_final.drop(['Open'], 1))


scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
X = scalerX.fit_transform(X)
Y = Y.reshape(-1, 1)
Y = scalerY.fit_transform(Y)
X_train_p, X_test, Y_train_p, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_p, Y_train_p, test_size=0.1, shuffle=False)

pred_days = 10
X_train_rs = []
y_train_rs = []
for i in range(60,(len(X_train)-pred_days)):
    X_train_rs.append(X_train[i-60:i])
    y_train_rs.append(Y_train[i:i+pred_days,0])
X_train_rs, y_train_rs = np.array(X_train_rs), np.array(y_train_rs)

X_val_rs = []
y_val_rs = []
for i in range(60,(len(X_val)-pred_days)):
    X_val_rs.append(X_val[i-60:i])
    y_val_rs.append(Y_val[i:i+pred_days,0])
X_val_rs, y_val_rs = np.array(X_val_rs), np.array(y_val_rs)

lstm_regressor = Sequential()

lstm_regressor.add(LSTM(units = 500, return_sequences=True, input_shape = [X_train_rs.shape[1],X_train_rs.shape[2]]))
lstm_regressor.add(Dropout(0.2))

lstm_regressor.add(LSTM(units = 500, return_sequences=True))
lstm_regressor.add(Dropout(0.2))

lstm_regressor.add(LSTM(units = 500, return_sequences=True))
lstm_regressor.add(Dropout(0.2))

lstm_regressor.add(LSTM(units = 500))
lstm_regressor.add(Dropout(0.2))

lstm_regressor.add(Dense(units = pred_days))

lstm_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

lstm_regressor.fit(X_train_rs, y_train_rs, epochs=7, batch_size=32, validation_data=(X_val_rs, y_val_rs))

#Predict and visualize
inputs = X[len(X)-len(X_test)-pred_days-60:-pred_days]
X_test_rs = []
for i in range(60, inputs.shape[0]):
    X_test_rs.append(inputs[i-60:i])
X_test_rs = np.array(X_test_rs)
predicted_stock_price = lstm_regressor.predict(X_test_rs)
predicted_stock_price = scalerY.inverse_transform(predicted_stock_price)

plt.plot(scalerY.inverse_transform(Y_test), color='red', label = 'Real Stock Price')
plt.plot(predicted_stock_price[:,9], color='blue', label='Predicted Stock Price')
plt.title('Illumina Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
x = predicted_stock_price[:,9]