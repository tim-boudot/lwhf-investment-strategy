'''
Implementation of Covariance Matrices using pypfopt
'''
from pypfopt import risk_models

def est_covariance(df_returns, method_cov, log_returns, returns_data = True):
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
        covariance = risk_models.risk_matrix(df_returns, method_cov, log_returns, returns_data = True)
        #check if covariance is positive semi-definite and fix- it if not!
        covariance = risk_models.fix_nonpositive_semidefinite(covariance, fix_method='spectral')


    return covariance
