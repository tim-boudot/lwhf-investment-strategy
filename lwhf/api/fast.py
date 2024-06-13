import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lwhf.ml_logic.backtesting import get_data, features_from_data, initialize_model_LSTM, fitting_model, predicting, making_portfolio, portfolio_returns, backtesting
from lwhf.portfolio.backtest import BackTester
import time

app = FastAPI()

#load model if existing
#app.state.model = load_model()

#TODO: Handle the saving and loading of model on GCloud
#TODO: Change backtesting function to make it more flexible

@app.get("/")
def root():
    return {'Le Wagon Hedge Fund Recommendation': 'Purchase Shiba Inu Coin'}  # YOUR CODE HERE


#uvicorn fast:app --reload --port 8205
# @app.get("/predict")
# def backtesting_demo(as_of_date: str, n_periods:int):
#     port_return, weekly_returns, cleaned_weigths = backtesting(as_of_date,n_periods)
#     return {'total return':port_return,
#             'weekly_returns':weekly_returns,
#             'latest_portfolio':cleaned_weigths}


@app.get("/backtest")
def final_backtest(as_of_date: str, n_periods:int):
    # time.sleep(15)
    print('I start now')
    bt = BackTester(as_of_date, n_periods)
    print('Initialized the class')
    bt.get_all_data()
    print('Got the data')
    bt.train_model()
    print('trained the model, starting backtesting')
    port_return, weekly_returns, clean_weights = bt.backtest()
    return {'total return':port_return,
            'weekly_returns':weekly_returns,
            'latest_portfolio':clean_weights}
