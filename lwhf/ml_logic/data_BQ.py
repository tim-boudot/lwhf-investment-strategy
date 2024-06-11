
from google.cloud import bigquery
from lwhf.params import *

def get_all_data():
    '''
    Get all data available from GOOGLE BIG QUERY
    '''
    query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{DATASET}.{TABLE}
    """
    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df

#TODO: ADD OTHER TABLES FOR GRANULARITY OF DATA
def get_data(start_date, end_date, timestep_data = 'W', project = GCP_PROJECT, dataset = DATASET, table = TABLE):

    '''
    Get data available from start_date to end_date with 'timestep_data' granulairity from GOOGLE BIG QUERY
    '''

    #time has to be in format YYYY-MM-DD
    assert len(start_date) == 10
    assert len(end_date) == 10

    # if  timestep_data == 'W':
    #     TABLE = "SP500_Historical_Weekly"

    # if  timestep_data == 'W':

    query = f"""
    SELECT *
    FROM {project}.{dataset}.{table}
    WHERE (DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}')
    """

    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    return df
