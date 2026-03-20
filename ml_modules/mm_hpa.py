# predictor/ml_engine/mm_hpa.py
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class MMHPA:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.linear_models = {}
        self.nonlinear_model = None
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()

    def _arima_predictions(self, prices, order=(2, 1, 1), window=60):
        predictions = np.zeros(len(prices))
        for i in range(len(prices)):
            if i < window:
                predictions[i] = prices[i]
            else:
                try:
                    train = prices[max(0, i - window):i]
                    model = ARIMA(train, order=order)
                    fitted = model.fit()
                    predictions[i] = fitted.forecast(steps=1)[0]
                except:
                    predictions[i] = prices[i - 1]
        return predictions

    def _linear_regression_predictions(self, prices, window=30):
        predictions = np.zeros(len(prices))
        for i in range(len(prices)):
            if i < window:
                predictions[i] = prices[i]
            else:
                X_lr = np.arange(window).reshape(-1, 1)
                y_lr = prices[i - window:i]
                lr = LinearRegression()
                lr.fit(X_lr, y_lr)
                predictions[i] = lr.predict([[window]])[0]
        return predictions

    def _build_lstm(self, n_features):
        model = Sequential([
            LSTM(64, return_sequences=True,
                 input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError()
        )
        return model

    def fit_predict(self, close_prices, features_df=None, epochs=50):
        print("\n    === MM-HPA: Multi-Model Hybrid Prediction ===")

        close_prices = np.array(close_prices).flatten()

        print("      ARIMA predictions...")
        arima_pred = self._arima_predictions(close_prices)

        print("      Linear Regression predictions...")
        lr_pred = self._linear_regression_predictions(close_prices)

        linear_features = np.column_stack([arima_pred, lr_pred])

        if features_df is not None:
            combined = np.column_stack([features_df, linear_features])
        else:
            combined = linear_features

        combined_scaled = self.scaler.fit_transform(combined)
        prices_scaled = self.price_scaler.fit_transform(
            close_prices.reshape(-1, 1)
        ).flatten()

        X, y = [], []
        for i in range(len(combined_scaled) - self.sequence_length):
            X.append(combined_scaled[i:i + self.sequence_length])
            y.append(prices_scaled[i + self.sequence_length])
        X = np.array(X)
        y = np.array(y)

        print("      Training LSTM...")
        self.nonlinear_model = self._build_lstm(combined_scaled.shape[1])
        self.nonlinear_model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        predictions_scaled = self.nonlinear_model.predict(X, verbose=0).flatten()
        predictions = self.price_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        return predictions, combined_scaled, prices_scaled