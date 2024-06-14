import datetime
from lwhf.data.biqquery import BigQueryData, get_Xy
from lwhf.params import GCP_PROJECT, DATASET, TABLE
from lwhf.models.lstm import SimpleLSTM
from pypfopt import risk_models, EfficientFrontier
import pandas as pd
import riskfolio as rp
import numpy as np


def get_total_return(weekly_returns):
    total_return = 1
    for weekly_return in weekly_returns:
        total_return *= (1 + weekly_return)
    return total_return - 1

def estimate_covariance(df_returns, method_cov, log_returns = False, returns_data = True):
    '''
    Returns estimate of the covariance

    Input:
    * df_returns of ADJUSTED Closing Prices
    * returns_data = False: dataframe containign prices, default is True => datafram is of returns
    * method_cov = [ 'pandas', 'sample_cov', 'semicovariance', 'exp_cov', 'ledoit_wolf', 'ledoit_wolf_constant_variance', 'ledoit_wolf_single_factor', 'ledoit_wolf_constant_correlation', 'oracle_approximating']
    * log_returns: specify wether matrix ig log return or returns
    '''

    if method_cov == 'pandas':
        covariance = df_returns.cov()

    else:
        #sample covariance matrix using method_cov
        covariance = risk_models.risk_matrix(prices = df_returns, method = method_cov, returns_data = True)
        #check if covariance is positive semi-definite and fix- it if not!
        covariance = risk_models.fix_nonpositive_semidefinite(covariance, fix_method='spectral')

    return covariance

# depreciated
def make_portfolio(tickers,expected_returns, cov_df):
    ef = EfficientFrontier(expected_returns,cov_df)#, solver='ECOS') #Had to change the solver to ECOS as the other wouldn't work. Look into this.
    ef.tickers = tickers
    raw_weights = ef.max_sharpe(risk_free_rate=0.001)
    cleaned_weights = ef.clean_weights()
    return pd.DataFrame(list(cleaned_weights.items()), columns=['ticker','weight']).set_index('ticker')


class BackTester:
    def __init__(self,
                 as_of_date,
                 n_periods):
        self.as_of_date = as_of_date
        self.n_periods = n_periods

    def get_all_data(self):
        bq = BigQueryData(GCP_PROJECT, DATASET, TABLE)
        bq.get_data('2016-01-04', self.as_of_date)
        bq.get_prices()
        bq.get_returns()
        self.bq = bq
        return self.bq

    def train_model(self):
        as_of = datetime.datetime.strptime(self.as_of_date, '%Y-%m-%d').date()
        starting_point = as_of - datetime.timedelta(days=7 * self.n_periods)
        returns_df = self.bq.returns
        train_df = returns_df[returns_df.index.date < starting_point]
        train_X, train_y = get_Xy(train_df)
        self.model = SimpleLSTM(train_X, train_y)
        self.model.initialize_model()
        self.model.fit_model()

    def build_portfolio(self):
        as_of = datetime.datetime.strptime(self.as_of_date, '%Y-%m-%d').date()
        starting_point = as_of - datetime.timedelta(days=7 * self.n_periods)
        returns_df = self.bq.returns
        pred_df = returns_df[returns_df.index.date <= starting_point]
        pred_X, _ = get_Xy(pred_df)
        y_pred = self.model.predict(pred_X)
        cov_df = pred_df.cov()
        port = rp.Portfolio(returns=pred_df, nea=8)

        if np.all(y_pred < 0):
            port.mu = pred_df.mean().values.reshape(-1)
        else:
            port.mu = y_pred.reshape(-1)
        port.cov = cov_df

        clean_weights = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0).round(5)

        exp_return = (y_pred * clean_weights.round(5).values).sum()
        exp_variance = np.multiply(np.dot(clean_weights,clean_weights.T),cov_df).sum().sum()

        return exp_return, exp_variance, clean_weights

    def backtest(self):
        as_of = datetime.datetime.strptime(self.as_of_date, '%Y-%m-%d').date()
        starting_point = as_of - datetime.timedelta(days=7 * self.n_periods)
        returns_df = self.bq.returns

        # port_return = 1
        portfolio_returns = []
        market_returns = []
        weekly_weights = []

        # have uniform weights to start with
        ticker_names = list(returns_df.columns)
        num_tickers = len(ticker_names)
        uniform_weights = np.full(num_tickers, 1/num_tickers)
        clean_weights = pd.DataFrame(data={'weights': uniform_weights}, index=ticker_names)
        zero_weights = pd.DataFrame(data={'weights': np.zeros(num_tickers)}, index=ticker_names)

        while starting_point < as_of:
            one_week_ahead = starting_point + datetime.timedelta(days=7)
            print(f'----- Predicting for week {starting_point} to {one_week_ahead}')

            pred_df = returns_df[returns_df.index.date < starting_point]
            pred_X, _ = get_Xy(pred_df)
            y_pred = self.model.predict(pred_X)
            cov_df = estimate_covariance(pred_df, 'pandas')

            #print(f' -- shape of pred_df: {pred_df.shape}')

            prices_df = self.bq.prices
            week_df = prices_df[prices_df.index.date >= starting_point]
            week_df = week_df[week_df.index.date <= one_week_ahead]

            if np.all(y_pred < 0):
                print(' -- all returns negative')
                starting_point += datetime.timedelta(days=7)
                ret = week_df.iloc[-1] / week_df.iloc[0] - 1
                market_return = (uniform_weights * ret).sum()
                market_returns.append(market_return)
                portfolio_returns.append(0)
                weekly_weights.append(zero_weights)
                continue

            port = rp.Portfolio(returns=pred_df, nea=8)
            port.mu = y_pred.reshape(-1)
            port.cov = cov_df
            clean_weights = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0)

            #print(f' -- optimized weights: {clean_weights.shape}')



            # print(f' -- shape of pred_X: {pred_X.shape}')
            # print(f' -- shape of y_pred: {y_pred.shape}')
            # print(f' -- shape of cov_df: {cov_df.shape}')
            # print(f' -- shape of prices_df: {prices_df.shape}')
            # print(f' -- shape of week_df: {week_df.shape}')

            ret = week_df.iloc[-1] / week_df.iloc[0] - 1
            weekly_return = (clean_weights.weights * ret).sum()
            market_return = (uniform_weights * ret).sum()

            portfolio_returns.append(weekly_return)
            market_returns.append(market_return)
            weekly_weights.append(clean_weights)
            # port_return *= (1+weekly_return)
            starting_point += datetime.timedelta(days=7)

        # port_return -= 1

        return market_returns, portfolio_returns, weekly_weights
