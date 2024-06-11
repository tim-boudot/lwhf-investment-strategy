'''
Implementation of Covariance matrices from portfolio Lab
'''

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


'''
df: input of stock data with columns = tickers and rows, price
'''
df = pd.read_csv("stock_data.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)


def est_covariance(df_returns, method_cov ='pandas', returns_data=True):
    '''
    Returns estimate of the covariance

    Input:
    * df_returns of Adjusted Closing Prices
    * returns_data = False: dataframe containign prices, default returns_data=True df is of returns
    * method_cov = [pandas, sample_cov, semicovariance, exp_cov, ledoit_wolf, ledoit_wolf_constant_variance, ledoit_wolf_single_factor, ledoit_wolf_constant_correlation, oracle_approximating]
    '''

    if method_cov == 'pandas':
        covariance = df_returns.cov()

    else:
        #annualised sample covariance matrix
        covariance = risk_models.risk_matrix(df_returns, method_cov, returns_data = True)

    return covariance
