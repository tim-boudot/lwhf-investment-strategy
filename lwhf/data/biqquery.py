from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import numpy as np
from lwhf.params import QUERIED_CACHE_BUCKET, QUERIED_CACHE_LOCAL
import os

def save_to_bucket(filename):
    '''
    Save a file to the bucket
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(QUERIED_CACHE_BUCKET)
    blob = bucket.blob(filename)
    if blob.exists():
        blob.delete()
    blob = bucket.blob(filename)
    blob.upload_from_filename(f'{QUERIED_CACHE_LOCAL}/{filename}')
    #blob.upload_from_filename(QUERIED_CACHE_LOCAL)
    print("✅ Model saved to GCS")
    return None

def check_bucket(filename):
    '''
    Check if a file exists in the bucket
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(QUERIED_CACHE_BUCKET)
    blob = bucket.blob(filename)

    if blob.exists():
        print(f'✅ Found {filename} in the bucket.')
        blob.download_to_filename(f'{QUERIED_CACHE_LOCAL}/{filename}')
        #blob.download_to_filename(QUERIED_CACHE_LOCAL)
    return None

def check_local(filename):
    '''
    Check if a file exists locally
    '''
    file_path = f'{QUERIED_CACHE_LOCAL}/{filename}'
    if os.path.exists(file_path):
        print(f'✅ Found {filename} in the local cache.')
        df = pd.read_csv(file_path)
        return df
    else:
        return None

def query_all_data(project, dataset, table):
    '''
    Get all data available from GOOGLE BIG QUERY
    '''
    filename = f'{project}-{dataset}-{table}.csv'
    print(filename)

    # check if the file is in the local cache
    df = check_local(filename)
    if df is not None:
        return df
    else:
        # check if the file is in the bucket
        check_bucket(filename)
        df = check_local(filename)
        if df is not None:
            return df

    query = f"""
    SELECT *
    FROM {project}.{dataset}.{table}
    """
    client = bigquery.Client(project=project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    'TODO: this is a temporary fix, using only local cache for now. Need to implement bucket cache.'
    # save the file to the bucket
    df.to_csv(f'{QUERIED_CACHE_LOCAL}/{filename}', index=False)
    #save_to_bucket(filename)

    return df

def query_between_dates(project, dataset, table, start_date, end_date):
    '''
    Get data available from start_date to end_date with 'timestep_data' granulairity from GOOGLE BIG QUERY
    '''
    if len(start_date) != 10:
        raise ValueError('start_date has to be in format YYYY-MM-DD')
    if len(end_date) != 10:
        raise ValueError('end_date has to be in format YYYY-MM-DD')

    filename = f'{project}-{dataset}-{table}-{start_date}-{end_date}.csv'
    print(filename)

    # check if the file is in the local cache
    df = check_local(filename)
    if df is not None:
        return df
    else:
        # check if the file is in the bucket
        check_bucket(filename)
        df = check_local(filename)
        if df is not None:
            return df

    query = f"""
    SELECT *
    FROM {project}.{dataset}.{table}
    WHERE (DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}')
    """

    client = bigquery.Client(project=project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    'TODO: this is a temporary fix, using only local cache for now. Need to implement bucket cache.'
    # save the file to the bucket
    df.to_csv(f'{QUERIED_CACHE_LOCAL}/{filename}', index=False)
    #save_to_bucket(filename)

    return df

def get_Xy(df):
    X = df.iloc[:-1].to_numpy().T
    X = np.expand_dims(X, axis=2)
    y = df.iloc[-1].to_numpy()
    return X, y

class BigQueryData:
    def __init__(self,
                 project,
                 dataset,
                 table,
                 index_col = 'timestamp',
                 value_col = 'close',
                 ticker_col = 'symbol'):

        self.project = project
        self.dataset = dataset
        self.table = table
        self.index_col = index_col
        self.value_col = value_col
        self.ticker_col = ticker_col
        self.raw_data = None
        self.prices = None,
        self.returns = None

    def get_data(self, start_date=None, end_date=None):
        '''
        Get data available from start_date to end_date with 'timestep_data' granulairity from GOOGLE BIG QUERY
        '''
        if start_date is None and end_date is None:
            df = query_all_data(self.project, self.dataset, self.table)
        else:
            df =  query_between_dates(self.project, self.dataset, self.table, start_date, end_date)

        df[self.index_col] = pd.to_datetime(df[self.index_col])

        self.raw_data = df
        return df

    def get_prices_alpaca(self, handle_nulls = 'drop'):
        if self.raw_data is None:
            raise ValueError('No data available. Please run get_data() first.')

        df = self.raw_data[[self.index_col, self.ticker_col, self.value_col]]

        if handle_nulls == 'drop':
            # select tickers that have all data points
            max_count = df[self.ticker_col].value_counts().max()
            stocks_in_universe = df[self.ticker_col].value_counts().loc[df[self.ticker_col].value_counts()>max_count-1]
            df = df[df[self.ticker_col].isin(stocks_in_universe.index)]

        # pivot the data to get close price for each stock
        prices = df.pivot(index=self.index_col,
                          columns=self.ticker_col,
                          values=self.value_col)

        self.prices = prices
        return prices

    def get_prices(self, data_source = 'alpaca'):
        if data_source == 'alpaca':
            return self.get_prices_alpaca()
        else:
            raise ValueError('Invalid data source. Please use either "alpaca" or implement the another get_prices_* method.')

    def get_returns(self):
        if self.prices is None:
            raise ValueError('No prices available. Please run get_prices() first.')

        # we need to drop the first column as it will have NaN values
        returns = self.prices.pct_change().dropna()

        self.returns = returns
        return returns

    def get_Xy(self, based_on='returns'):
        if based_on == 'returns':
            if self.returns is None:
                raise ValueError('No returns available. Please run get_returns() first.')
            else:
                df = self.returns
        elif based_on == 'prices':
            if self.prices is None:
                raise ValueError('No prices available. Please run get_prices() first.')
            else:
                df = self.prices
        else:
            raise ValueError('Invalid value for based_on. Please use either "returns" or "prices".')

        X, y = get_Xy(df)

        print(f'Created X and y with shapes {X.shape} and {y.shape}')

        return X, y

    '''TODO: This is totally wrong, should have same dim as X'''
    def get_X_pred(self, based_on='returns'):
        '''
        This is just to check consistency, remove later
        '''
        'TODO: the reshape method is wrong, see above'
        return self.returns_df.to_numpy().reshape(self.returns_df.shape[1],
                                                  self.returns_df.shape[0],
                                                  1)
