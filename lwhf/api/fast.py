import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lwhf.ml_logic.backtesting import get_data, features_from_data, initialize_model_LSTM, fitting_model, predicting, making_portfolio, portfolio_returns, backtesting

app = FastAPI()

#TODO: Handle the saving and loading of model on GCloud
#TODO: Change backtesting function to make it more flexible

@app.get("/")
def root():
    return {'greeting': 'Hello'}  # YOUR CODE HERE

@app.get("/predict")
def backtesting_demo(as_of_date: str, n_periods:int):
    port_return, weekly_returns, cleaned_weigths = backtesting(as_of_date,n_periods)
    return {'total return':port_return,
            'weekly_returns':weekly_returns,
            'latest_portfolio':cleaned_weigths}
