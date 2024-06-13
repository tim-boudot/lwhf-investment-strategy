import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lwhf.ml_logic.backtesting import get_data, features_from_data, initialize_model_LSTM, fitting_model, predicting, making_portfolio, portfolio_returns, backtesting
from lwhf.portfolio.backtest import BackTester, get_total_return
import time, os, json
from lwhf.params import QUERIED_CACHE_LOCAL



app = FastAPI()

#load model if existing
#app.state.model = load_model()

#TODO: Handle the saving and loading of model on GCloud
#TODO: Change backtesting function to make it more flexible

@app.get("/")
def root():
    return {'Le Wagon Hedge Fund Recommendation': 'Purchase Bitcoin'}  # YOUR CODE HERE


#uvicorn fast:app --reload --port 8205
# @app.get("/predict")
# def backtesting_demo(as_of_date: str, n_periods:int):
#     port_return, weekly_returns, cleaned_weigths = backtesting(as_of_date,n_periods)
#     return {'total return':port_return,
#             'weekly_returns':weekly_returns,
#             'latest_portfolio':cleaned_weigths}


@app.get("/clear_api_cache")
def clear_api_cache():
    backtest_api_cache_folder = 'backtest_api_cache'
    full_path = os.path.join(QUERIED_CACHE_LOCAL, backtest_api_cache_folder)
    if os.path.exists(full_path):
        for file in os.listdir(full_path):
            os.remove(os.path.join(full_path, file))
        return {'message': 'Cache cleared'}
    return {'message': 'Cache was already empty'}

@app.get("/clear_data_cache")
def clear_data_cache():
    full_path = QUERIED_CACHE_LOCAL
    if os.path.exists(full_path):
        for file in os.listdir(full_path):
            os.remove(os.path.join(full_path, file))
        return {'message': 'Cache cleared'}

@app.get("/backtest")
def final_backtest(as_of_date: str, n_periods:int):

    backtest_api_cache_folder = 'backtest_api_cache'
    full_path = os.path.join(QUERIED_CACHE_LOCAL, backtest_api_cache_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    filename = f'{as_of_date}-{n_periods}.json'
    json_full_path = os.path.join(full_path, filename)
    if os.path.exists(json_full_path):
        print(f'✅ Found {filename} in the local cache.')
        # read the json file as a dictionary
        with open(json_full_path, 'r') as file:
            result = json.load(file)
        return result


    # time.sleep(15)
    print('I start now')
    bt = BackTester(as_of_date, n_periods)
    print('Initialized the class')
    bt.get_all_data()
    print('Got the data')
    bt.train_model()
    print('trained the model, starting backtesting')
    market_returns, portfolio_returns, weekly_weights = bt.backtest()
    total_portfolio_return = get_total_return(portfolio_returns)
    total_market_return = get_total_return(market_returns)
    weekly_weights = [week_df.to_dict()['weights'] for week_df in weekly_weights]
    final_weights = weekly_weights[-1]

    result = {
        'market_returns': market_returns,
        'total_market_return': total_market_return,
        'portfolio_returns': portfolio_returns,
        'total_portfolio_return': total_portfolio_return,
        'weekly_weights': weekly_weights,
        'final_weights': final_weights
    }
    # save the file as a json to full_path
    with open(json_full_path, 'w') as f:
        json.dump(result, f)

    return result
