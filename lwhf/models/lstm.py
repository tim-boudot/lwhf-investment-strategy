
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


class SimpleLSTM:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.input_shape = X_train.shape[1:]
        self.model = None

    def initialize_model(self):
        inputs = layers.Input(self.input_shape)
        #normalizer.adapt(X)
        model = Sequential()
        model.add(inputs)
        #TODO: normalizer breaks the model?
        #model.add(normalizer)
        model.add(layers.LSTM(units=20, activation='tanh'))
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dropout(0.1))
        #model.add(layers.LSTM(units=20, activation='linear'))
        model.add(layers.Dense(1, activation="linear"))

        # 2- Compilation
        model.compile(loss='mse',
                    optimizer='rmsprop', #adam
                    metrics=['mae']) # very high lr so we can converge with such a small datase

        self.model = model
        return model

    def fit_model(self):
        if self.model is None:
            raise ValueError('No model available. Please run initialize first.')

        es = EarlyStopping(
            patience=5,
            restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train,
            self.y_train.reshape(-1),
            validation_split=.2,
            batch_size=32,
            epochs=10,
            verbose=10,
            callbacks=[es])

        return self.history

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
