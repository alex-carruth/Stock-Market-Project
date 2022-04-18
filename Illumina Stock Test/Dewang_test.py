# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:29:06 2021

@author: acarr
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from illumina_combiner import ilmn_final

def prediction(ticker, inputs, s):
    def CSV_to_DF(ticker):
        # Importing the dataset
        file = str(ticker) + '_updated.csv'
        datasets = pd.read_csv(file)
        return datasets

    # Shifts Open prices down the dataframe column by 's' days
    dataset = CSV_to_DF(str(ticker))
    dataset['0'] = dataset['0'].shift(s)
    SIZE = len(dataset) - s
    dataset = dataset.tail(SIZE)
    
    
    
    def Equation(data):
        # Determine Inputs and Output
        y = data.iloc[:, 1].values
        X = data.iloc[:, 2:].values
        # Splitting the Dataset into the Training Set and Test Set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        np.set_printoptions(precision=2)

        r = [regressor.coef_, regressor.intercept_]
        return r

    def predict(set, input):
        coeff = Equation(set)[0]
        intercept = Equation(set)[1]
        predicted = 0
        for i in range(len(coeff)):
            predicted += coeff[i] * input[i]
        predicted += intercept
        return predicted

    return predict(dataset, inputs)

# Convert Updated_CSV to Pandas DataFrame
def CSV_to_DF(ticker):
        file = ticker + '_updated.csv'
        datasets = pd.read_csv(file)
        return datasets

# Returns percent error of an algorithm for any given day
def percentError(n, day, ticker):
        dataset = CSV_to_DF(ticker)
        inputs = dataset.iloc[day, 2:].values
        out1 = dataset.iloc[day + n, 1]
        out2 = prediction(ticker, inputs, n)
        out = ((out2 - out1) / (out1)) * 100
        return out

# Returns percent error in prediction over a certain range
def percentErrorOverRange(d1, d2, d, ticker):
        b = 0
        for i in range(d1, d2):
                a = percentError(d, i, ticker)
                b += abs(a)
        returnStr = round(b/(d2 - d1),3)
        return returnStr

def Error(n, ticker):
        return percentErrorOverRange(0,2566-(2*n),n, ticker)

#ilmn_final.to_csv('ilmn_2_updated.csv')

# Setting Coordinates with 'x' as Dates and 'y' as Error
x = range(1,40)
y = []
# Error Test for AMD
for i in range(1,40):
        y.append(Error(i, 'ilmn_2')) 
        
# Graphing Error
plt.plot(x, y) 
plt.xlabel('Days') 
plt.ylabel('Error') 
plt.title('Average Percent Error of Predictions') 
plt.show() 



