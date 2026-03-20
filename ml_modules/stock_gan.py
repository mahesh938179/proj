# predictor/ml_engine/stock_gan.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D,
    Flatten, BatchNormalization, LeakyReLU, Reshape
)
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


class StockGAN:
    def __init__(self, sequence_length, n_features, hyperparams=None):
        self.sequence_length = sequence_length
        self.n_features = n_features

        self.hyperparams = hyperparams or {
            'filters': 64,
            'dropout': 0.3,
            'kernel_size': 3,
            'padding': 'same',
            'batchnorm_momentum': 0.8,
            'lrelu_alpha': 0.2,
            'strides': 1,
            'cnn_lr': 0.0002,
            'lstm_lr': 0.001,
            'batch_size': 32,
            'lstm_units_1': 128,
            'lstm_units_2': 64,
        }

        self.generator = None
        self.discriminator = None
        self._build_generator()
        self._build_discriminator()

    def _build_generator(self):
        hp = self.hyperparams
        model = Sequential(name='Generator_LSTM')

        model.add(LSTM(
            units=int(hp['lstm_units_1']),
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(hp['dropout']))

        model.add(LSTM(
            units=int(hp['lstm_units_2']),
            return_sequences=True
        ))
        model.add(Dropout(hp['dropout']))

        model.add(LSTM(
            units=int(hp['lstm_units_2']) // 2,
            return_sequences=False
        ))
        model.add(Dropout(hp['dropout']))

        model.add(Dense(1, activation='linear'))

        model.compile(
            optimizer=Adam(learning_rate=hp['lstm_lr']),
            loss=tf.keras.losses.MeanSquaredError()
        )

        self.generator = model

    def _build_discriminator(self):
        hp = self.hyperparams
        model = Sequential(name='Discriminator_CNN')

        model.add(Reshape(
            (self.sequence_length + 1, 1),
            input_shape=(self.sequence_length + 1,)
        ))

        model.add(Conv1D(
            filters=int(hp['filters']),
            kernel_size=int(hp['kernel_size']),
            strides=int(hp['strides']),
            padding=hp['padding']
        ))
        model.add(BatchNormalization(momentum=hp['batchnorm_momentum']))
        model.add(LeakyReLU(alpha=hp['lrelu_alpha']))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(
            filters=int(hp['filters']) * 2,
            kernel_size=int(hp['kernel_size']),
            strides=int(hp['strides']),
            padding=hp['padding']
        ))
        model.add(BatchNormalization(momentum=hp['batchnorm_momentum']))
        model.add(LeakyReLU(alpha=hp['lrelu_alpha']))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=hp['lrelu_alpha']))
        model.add(Dropout(hp['dropout']))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=hp['cnn_lr']),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
        )

        self.discriminator = model

    def train(self, X_train, y_train, epochs=100, batch_size=None):
        if batch_size is None:
            batch_size = int(self.hyperparams['batch_size'])

        n_samples = len(X_train)
        n_batches = max(1, n_samples // batch_size)

        d_losses = []
        g_losses = []

        print(f"\n    Training Stock-GAN for {epochs} epochs...")
        print(f"    Samples: {n_samples}, Batch: {batch_size}")

        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0

            indices = np.random.permutation(n_samples)

            for batch_idx in range(n_batches):
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                actual_batch_size = len(X_batch)

                generated_prices = self.generator.predict(X_batch, verbose=0).flatten()

                real_seq = X_batch[:, :, 0].copy()
                real_input = np.column_stack([real_seq, y_batch.reshape(-1, 1)])
                fake_input = np.column_stack([real_seq, generated_prices.reshape(-1, 1)])

                real_labels = np.ones((actual_batch_size, 1)) * 0.9
                fake_labels = np.zeros((actual_batch_size, 1)) + 0.1

                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(real_input, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_input, fake_labels)
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

                self.discriminator.trainable = False
                g_loss = self.generator.train_on_batch(X_batch, y_batch)

                generated_new = self.generator.predict(X_batch, verbose=0).flatten()
                fake_new = np.column_stack([real_seq, generated_new.reshape(-1, 1)])
                d_pred = self.discriminator.predict(fake_new, verbose=0)
                adv_loss = -np.mean(np.log(d_pred + 1e-10))
                combined_loss = g_loss + 0.1 * adv_loss

                epoch_d_loss += d_loss
                epoch_g_loss += combined_loss

            avg_d = epoch_d_loss / n_batches
            avg_g = epoch_g_loss / n_batches
            d_losses.append(avg_d)
            g_losses.append(avg_g)

            if (epoch + 1) % 20 == 0:
                print(f"      Epoch {epoch + 1}/{epochs} - D: {avg_d:.6f}, G: {avg_g:.6f}")

        return d_losses, g_losses

    def predict(self, X):
        return self.generator.predict(X, verbose=0).flatten()