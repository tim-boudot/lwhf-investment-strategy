import pandas as pd
import numpy as np
import datetime as DT
import pandas as pd
from pypfopt import EfficientFrontier
from google.cloud import bigquery
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization
# from pypfopt import risk_models
# from pypfopt import expected_returns
import os

gcp_project = os.environ['GCP_PROJECT']

def get_all_data():
    # start_date=datetime.strptime(start_date,'%Y-%m-%d')
    # end_date=datetime.strptime(end_date,'%Y-%m-%d')
    PROJECT = "le-wagon-hedge-fund"
    DATASET = "data_alpaca_20240604"
    TABLE = "SP500_Historical_Weekly"
    query = f"""

    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
    """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

def get_data(start_date, end_date):
    # start_date=datetime.strptime(start_date,'%Y-%m-%d')
    # end_date=datetime.strptime(end_date,'%Y-%m-%d')
    PROJECT = "le-wagon-hedge-fund"
    DATASET = "data_alpaca_20240604"
    TABLE = "SP500_Historical_Weekly"
    query = f"""

    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
    WHERE (DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}')
    """
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

def features_from_data(df):
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

    return X, y, X_pred, returns_df.cov(), list(returns_df.columns)

def initialize_model_LSTM(X):
    # 1- RNN Architecture
    normalizer = Normalization()
    normalizer.adapt(X)
    model = Sequential()
    model.add(normalizer)
    model.add(layers.LSTM(units=20, activation='tanh'))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))

    # 2- Compilation
    model.compile(loss='mse',
                optimizer='rmsprop',
                metrics=['mae']) # very high lr so we can converge with such a small dataset

    return model

def fitting_model(X,y):
    model = initialize_model_LSTM(X)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X, y.reshape(-1,), validation_split=.2, batch_size=32, epochs=20, verbose=1, callbacks=[es])
    return model

def predicting(X, model):
    y_pred = model.predict(X)
    return y_pred

def making_portfolio(tickers,expected_returns, cov_df):
    ef = EfficientFrontier(expected_returns,cov_df, solver='ECOS') #Had to change the solver to ECOS as the other wouldn't work. Look into this.
    ef.tickers = tickers
    raw_weights = ef.max_sharpe(risk_free_rate=0.001)
    cleaned_weights = ef.clean_weights()
    return pd.DataFrame(list(cleaned_weights.items()), columns=['ticker','weight']).set_index('ticker')

def portfolio_returns(weights: pd.DataFrame, start_date: str, end_date: str):
    # Finding the returns for all stocks between start and end date
    df=get_data(start_date,end_date)
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

def backtesting(as_of_date, n_periods, period_type='W'):
    as_of = DT.datetime.strptime(as_of_date, '%Y-%m-%d').date()
    starting_point = as_of - DT.timedelta(days=7 * n_periods)
    starting_point_str = f'{starting_point.year}-{starting_point.month:02d}-{starting_point.day:02d}'
    port_return = 1
    weekly_returns = []

    # Training the model with data until the starting point

    df = get_data('2016-01-04',as_of_date)
    df = df[df.timestamp.apply(lambda x: DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00').date())<starting_point]

    X, y, X_pred, cov_df, tickers = features_from_data(df)
    model = fitting_model(X,y)

    # Calculating portfolio returns
    while starting_point < as_of:
        one_week_ahead = starting_point + DT.timedelta(days=7)
        week_start_str = f'{starting_point.year}-{starting_point.month:02d}-{starting_point.day:02d}'
        week_end_str = f'{one_week_ahead.year}-{one_week_ahead.month:02d}-{one_week_ahead.day:02d}'

        df = df[df.timestamp.apply(lambda x: DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00').date())<starting_point]
        X, y, X_pred, cov_df, tickers = features_from_data(df)
        y_pred = predicting(X_pred, model)
        cleaned_weights = making_portfolio(tickers,y_pred.reshape(-1), cov_df)
        weekly_return = portfolio_returns(cleaned_weights,week_start_str,week_end_str)
        weekly_returns.append(weekly_return)
        port_return *= (1+weekly_return)
        starting_point += DT.timedelta(days=7)

    port_return -= 1

    return port_return, weekly_returns, cleaned_weights
