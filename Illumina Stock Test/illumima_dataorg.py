# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:41:48 2021

@author: acarr
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as web
from datetime import date
from dateutil.relativedelta import relativedelta
from finta import TA
import time

INDICATORS = ['STOCH', 'ADL', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

def _get_indicator_data(data):
    
    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    
    return data

def get_needed_data(ticker, start, end):
    start = start
    end = end
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Open']    
    retFrame = pd.DataFrame(info)
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Close']
    retFrame['Close'] = info
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Adj Close']
    retFrame['Adj Close'] = info
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Volume']
    retFrame['Volume'] = info
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['High']
    retFrame['High'] = info
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Low']
    retFrame['Low'] = info
    info = web.DataReader('spy', data_source='yahoo', start=start, end=end)['Adj Close']
    retFrame['SPY'] = info
    return retFrame

def compute_RSI(data, time_window):
    diff = data.diff(1).dropna()
    
    #Preserving dimensions
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    #Positive difference
    up_chg[diff > 0] = diff[diff > 0]
    
    #Negative difference
    down_chg[diff < 0] = diff[diff < 0]
    
    #setting up com=time_window-1 so decay is alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def compute_MACD(data):
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    macd_signal = macd-exp3
    return macd_signal

#not sure if this is calculated correctly
def compute_WilliamsR(dataH, dataL, close):
    maxW = dataH.rolling(window=14).max()
    minW = dataL.rolling(window=14).min()
    williamsR = (maxW - close)/(maxW - minW)
    return williamsR

def compute_Volatility(data):
    returns = np.log(data/(data.shift(1)))
    vol = returns.rolling(window=252).std() * np.sqrt(252)
    return vol
    
today = date.today()
prev = today - relativedelta(years=5)
dt = today.strftime("%d/%m/%Y")
pdt = prev.strftime("%d/%m/%Y")

#Ensuring Connection
connected = False
while not connected:
    try:
        ilmn = get_needed_data('ilmn',pdt,dt)
        qgen = get_needed_data('qgen',pdt,dt)
        tmo = get_needed_data('tmo',pdt,dt)
        connected = True
    except Exception as e:
        print("type error: " + str(e))
        time.sleep(5)
        pass


for item in (ilmn,qgen,tmo):
    #The Moving Average
    item['30 Day MA'] = item['Adj Close'].rolling(window=20).mean()
    #Standard Deviation
    #You can set ddof=0 in std() to make it population instead of sample
    item['30 Day STD'] = item['Adj Close'].rolling(window=20).std()
    #Bollinger Band
    item['Upper Band'] = item['30 Day MA'] + (item['30 Day STD'] * 2)
    item['Lower Band'] = item['30 Day MA'] - (item['30 Day STD'] * 2)
    
    #Relative Strength Index
    item['RSI'] = compute_RSI(item['Adj Close'], 14)

    #Moving Average Convergence Divergence
    item['MACD'] = compute_MACD(item['Close'])
    
    #4 Week High
    item['4WH'] = item['High'].rolling(window=20).max()
    
    #4 Week Low
    item['4WL'] = item['Low'].rolling(window=20).min()
    
    #Williams %R
    item['WPR'] = compute_WilliamsR(item['High'],item['Low'],item['Close'])
    
    #Range
    item['Range'] = item['Open'] - item['Close']
    
    #Volatility    
    #item['Volatility'] = compute_Volatility(item['Close'])
    

    
    
#Stochastic oscillator, accumalation-distribution line, market momentum,
#money flow index, rate of change, on balance volume, commodity channel
#index, ease of movement, and vortext indicator
ilmn = _get_indicator_data(ilmn)
qgen = _get_indicator_data(qgen)
tmo = _get_indicator_data(tmo)

#Deleting unnecessary columns
labels = ['High', 'Low', 'Close']
ilmn = ilmn.drop(columns=labels, axis=1)
qgen = qgen.drop(columns=labels, axis=1)
tmo = tmo.drop(columns=labels, axis=1)
ilmn = ilmn[20:1259]
    