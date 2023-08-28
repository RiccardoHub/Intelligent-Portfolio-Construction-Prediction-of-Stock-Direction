import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

tickers='AAPL'
start = '2000-06-02'
end = '2016-04-26' 
aapl_data = yf.download(tickers, start=start, end=end)[['Adj Close', 'Volume', 'High', 'Low', 'Close']]

def exponential_smoothing(price_series: [pd.DataFrame, list], alpha = 0.2):
    smoothed_series = price_series.ewm(alpha=alpha).mean()

    return smoothed_series

def OnBalanceVolume(price_volume_series):
    price_volume_series['Daily_Price_Change'] = price_volume_series['Smoothed_adj_close'].diff()
    price_volume_series['Direction'] = price_volume_series['Daily_Price_Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    OBV = (price_volume_series['Direction'] * price_volume_series['Volume']).cumsum()

    return OBV

def StochasticOscillator(price_low_high_series: pd.DataFrame, K_high: int = 14, K_low: int = 3):
    price_low_high_series['L14'] = price_low_high_series['Low'].rolling(window=K_high).min()
    price_low_high_series['H14'] = price_low_high_series['High'].rolling(window=K_high).max()
    price_low_high_series['%K'] = 100*((price_low_high_series['Smoothed_adj_close'] - price_low_high_series['L14']) / (price_low_high_series['H14'] - price_low_high_series['L14']))
    price_low_high_series['%D'] = price_low_high_series['%K'].rolling(window=K_low).mean()

    return price_low_high_series[['%K', '%D']]

def MACD(price_series: [pd.DataFrame, list], short_period: int = 12, long_period: int = 26, signal_period: int = 9):
    MA_Fast = price_series['Smoothed_adj_close'].ewm(span=12,min_periods=long_period).mean()
    MA_Slow = price_series['Smoothed_adj_close'].ewm(span=26,min_periods=26).mean()
    MACD = MA_Fast - MA_Slow
    Signal = MACD.ewm(span=signal_period, adjust=False).mean()
    MACD = pd.concat([MACD.rename('MACD'), Signal.rename('Signal')], axis=1)

    return MACD

def categorical_price(price_series: [pd.DataFrame, list], forecast_period: int):
    price = np.sign(np.log(price_series['Smoothed_adj_close'].shift(-forecast_period)/price_series['Smoothed_adj_close']))

    return price

def prepare_data(raw_data: pd.DataFrame, forecast_period = range(1, 31)):
    raw_data = copy.deepcopy(raw_data)

    raw_data['Smoothed_adj_close'] = exponential_smoothing(raw_data['Adj Close'])
    raw_data['OBV'] = OnBalanceVolume(raw_data)
    raw_data[['%K', '%D']] = StochasticOscillator(raw_data)
    raw_data[['MACD', 'Signal']]= MACD(raw_data)
    for day in forecast_period:
        raw_data[f'Categorical_p_{day}'] = categorical_price(raw_data, day)

    return raw_data

def RFClassifier(pre_processed_data: pd.DataFrame, independent_variables: list, dependent_variable: str):
    pre_processed_data = copy.deepcopy(pre_processed_data)
    pre_processed_data = pre_processed_data.dropna()

    X = pre_processed_data[independent_variables]
    y = pre_processed_data[dependent_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)

    Accuracy = metrics.accuracy_score(y_test, prediction)
    Precision = metrics.average_precision_score(y_test, prediction)
    Recall = metrics.recall_score(y_test, prediction)
    F1_score = metrics.f1_score(y_test, prediction)

    return Accuracy, Precision, Recall, F1_score


print('hi')