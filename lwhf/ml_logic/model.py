
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from lwhf.params import *

from lwhf.ml_logic.data_GCS import save_model_GCS, check_model_GCS


def initialize_model_LSTM(X):
    # 1- RNN Architecture

    #normalizer = layers.Normalization()
    inputs = layers.Input(shape=X.shape[1:])
    #normalizer.adapt(X)
    model = Sequential()
    model.add(inputs)
    #TODO: normalizer breaks the model?
    #model.add(normalizer)
    model.add(layers.LSTM(units=20, activation='linear'))
    model.add(layers.Dense(10, activation="linear"))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(units=20, activation='linear'))
    model.add(layers.Dense(1, activation="linear"))

    # 2- Compilation
    model.compile(loss='mse',
                optimizer='rmsprop', #adam
                metrics=['mae']) # very high lr so we can converge with such a small datase

    return model


def fitting_model(X,y, start_date, end_date, timestep_data = 'W', type_model = 'LSTM'):

    model_name = f'{type_model}/{start_date}/{end_date}/{timestep_data}'

    #GET MODEL FROM GCS IF EXISTS
    if check_model_GCS(model_name):
        model = load_model(LOCAL_MODEL_PATH) #tensorflow load_model function
        return model

    #INITIALISE LSTM MODEL
    if type_model == 'LSTM':
        model = initialize_model_LSTM(X)
        es = EarlyStopping(patience=5, restore_best_weights=True)
        history = model.fit(X, y.reshape(-1), validation_split=.2, batch_size=32, epochs=10, verbose=10 ,callbacks=[es])

    #SAVING MODEL LOCALLY - note: only last one fitted is sr
    model.save(LOCAL_MODEL_PATH)
    print("✅ Model saved locally")

    #ESAVING EVERY MODEL FITTED TO GCS
    save_model_GCS(model, model_name)

    return model

def predicting(X, model):
    y_pred = model.predict(X)
    return y_pred
