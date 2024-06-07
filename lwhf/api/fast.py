import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lwhf.ml_logic.backtesting import get_data, features_from_data, initialize_model_LSTM, fitting_model, predicting, making_portfolio, portfolio_returns, backtesting

app = FastAPI()

#TODO: Handle the saving and loading of model on GCloud

@app.get("/")
def root():
    return {'greeting': 'Hello'}  # YOUR CODE HERE

@app.get("/predict")
def backtesting_demo():
    port_return, weekly_returns = backtesting('2024-05-27',4)
    return {'total return':port_return}
