# predictor/ml_engine/autoencoders.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler


class StackedAutoencoder:
    def __init__(self, input_dim, encoding_dims=[128, 64, 32]):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.scaler = MinMaxScaler()
        self.autoencoder = None
        self.encoder = None
        self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = input_layer

        for i, dim in enumerate(self.encoding_dims):
            encoded = Dense(dim, activation='relu', name=f'encoder_{i}')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(0.2)(encoded)

        bottleneck = Dense(
            self.encoding_dims[-1] // 2, activation='relu', name='bottleneck'
        )(encoded)

        decoded = bottleneck
        for i, dim in enumerate(reversed(self.encoding_dims)):
            decoded = Dense(dim, activation='relu', name=f'decoder_{i}')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(0.2)(decoded)

        output_layer = Dense(self.input_dim, activation='sigmoid', name='output')(decoded)

        self.autoencoder = Model(input_layer, output_layer)
        self.autoencoder.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError()
        )

        self.encoder = Model(input_layer, bottleneck)

    def fit_transform(self, X, epochs=50, batch_size=32):
        X_scaled = self.scaler.fit_transform(X)
        self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        encoded_features = self.encoder.predict(X_scaled, verbose=0)
        return encoded_features

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)