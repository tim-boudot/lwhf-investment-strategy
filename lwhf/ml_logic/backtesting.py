from lwhf.ml_logic.data_GCS import save_model_GCS, check_model_GCS
from lwhf.ml_logic.data_BQ import get_all_data, get_data
from lwhf.ml_logic.model import initialize_model_LSTM, fitting_model, predicting
from lwhf.ml_logic.Cov import est_covariance

import datetime as DT
from pypfopt import EfficientFrontier

import time
import pandas as pd
from sklearn.linear_model import LinearRegression
# from pypfopt import risk_models
# from pypfopt import expected_returns
import os
from lwhf.params import *


def features_from_data(df, method_cov, name_data):

    '''
    method_cov: method to estimate the covariance
    name_data: name of the data you are using (each dataset might have different features)
    '''

    if name_data == 'alpaca_stock_prices':

        # dataset which only uses Closing prices on alpaca weekly dataset

        time_df = df.pivot(index='timestamp',columns='symbol',values='close')
        returns_df = time_df.pct_change()#.dropna()
        # Removing all stocks that have more than 20 missing observations
        s = returns_df.isna().sum()>20
        to_ban = list(s[s].index)
        returns_df = returns_df[returns_df.columns[~returns_df.columns.isin(to_ban)]]
        # Imputing
        returns_df = returns_df.fillna(returns_df.mean())

        X = returns_df.iloc[:-1]
        y = returns_df.iloc[-1]

        X = X.to_numpy().reshape(X.shape[1],X.shape[0],1)
        y = y.to_numpy()

        X_pred = returns_df.to_numpy()
        X_pred = X_pred.reshape(X_pred.shape[1],X_pred.shape[0],1)

        #change method for another estimation of the covariance
        covariance = est_covariance(returns_df,  method_cov,  log_returns=False, returns_data=True)

    return X, y, X_pred, covariance, list(returns_df.columns)

def making_portfolio(tickers,expected_returns, cov_df):
    ef = EfficientFrontier(expected_returns,cov_df, solver='ECOS') #Had to change the solver to ECOS as the other wouldn't work. Look into this.
    ef.tickers = tickers
    raw_weights = ef.max_sharpe(risk_free_rate=0.001)
    cleaned_weights = ef.clean_weights()
    return pd.DataFrame(list(cleaned_weights.items()), columns=['ticker','weight']).set_index('ticker')

def portfolio_returns(weights: pd.DataFrame, start_date: str, end_date: str, timestep_data):
    # Finding the returns for all stocks between start and end date
    df = get_data(start_date, end_date,  timestep_data) # add this to change from tables etc project = GCP_PROJECT, dataset = DATASET, table = TABLE
    time_df = df.pivot(index='timestamp',columns='symbol',values='close')

    #Resetting index for the time_df
    time_df['clean_date']=time_df.index
    time_df['clean_date']=time_df['clean_date'].apply(lambda x: DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))\
        .apply(lambda x: f'{x.year}-{x.month:02d}-{x.day:02d}')
    time_df = time_df.set_index('clean_date')

    ret = time_df.loc[f'{end_date}']/time_df.loc[f'{start_date}']-1

    # Calculating portfolio return
    port_return = (weights.weight * ret).sum()

    return port_return

#TODO: Make this code more efficient by not querying every time but rather saving data locally while running

def backtesting(as_of_date, n_periods, timestep_data, name_data, method_cov = 'exp_cov'):

    '''
    name_data: name of the data depending on features you will use etc
        1. 'alpaca_stock_prices': features is only closing price, using weekly alpaca data
    method_cov: way to estimate the covariance
    as_of_date: until what time we will get the data
    n_periods: how many period we want to backtet on
    timestep_data: timestep of data we will use

    '''

    as_of = DT.datetime.strptime(as_of_date, '%Y-%m-%d').date()
    starting_point = as_of - DT.timedelta(days=7 * n_periods)
    starting_point_str = f'{starting_point.year}-{starting_point.month:02d}-{starting_point.day:02d}'
    port_return = 1
    weekly_returns = []

    # Training the model with data until the starting point
    print('Getting data from BQ')
    df = get_data('2016-01-04', as_of_date, timestep_data)

    df = df[df.timestamp.apply(lambda x: DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00').date())<starting_point]

    #dates on which data will be trained
    start_date = list(df.timestamp)[0]
    end_date = list(df.timestamp)[-1]

    X, y, X_pred, cov_df, tickers = features_from_data(df, method_cov, name_data)

    print(cov_df.head(10))

    '''

    model = fitting_model(X,y, start_date, end_date, timestep_data, name_data, type_model = 'LSTM')


    # Calculating portfolio returns
    while starting_point < as_of:
        one_week_ahead = starting_point + DT.timedelta(days=7)
        week_start_str = f'{starting_point.year}-{starting_point.month:02d}-{starting_point.day:02d}'
        week_end_str = f'{one_week_ahead.year}-{one_week_ahead.month:02d}-{one_week_ahead.day:02d}'

        df = df[df.timestamp.apply(lambda x: DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00').date())<starting_point]
        X, y, X_pred, cov_df, tickers = features_from_data(df, method_cov, name_data)
        y_pred = predicting(X_pred, model)
        cleaned_weights = making_portfolio(tickers,y_pred.reshape(-1), cov_df)
        weekly_return = portfolio_returns(cleaned_weights,week_start_str,week_end_str)
        weekly_returns.append(weekly_return)
        port_return *= (1+weekly_return)
        starting_point += DT.timedelta(days=7)

    port_return -= 1

    return port_return, weekly_returns, cleaned_weights
    '''



if __name__ == '__main__':
    backtesting('2024-05-27', 2, timestep_data='W')
