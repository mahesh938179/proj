# predictor/ml_engine/autoencoders.py
# Keras/TF are optional — only needed for training.
# For inference, an ONNX session is used instead.
import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None
    Model = None


class StackedAutoencoder:
    def __init__(self, input_dim, encoding_dims=[128, 64, 32]):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.scaler = MinMaxScaler()
        self.autoencoder = None
        self.encoder = None
        # ONNX inference session (set by load_onnx_encoder, used when encoder is None)
        self._onnx_sess = None
        self._onnx_inp_name = None
        if HAS_TF:
            self._build_model()

    def _build_model(self):
        """Build Keras model (only used during training)."""
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

    def load_onnx_encoder(self, onnx_path):
        """Load an ONNX encoder session for inference (no Keras needed)."""
        import onnxruntime as rt
        self._onnx_sess = rt.InferenceSession(str(onnx_path))
        self._onnx_inp_name = self._onnx_sess.get_inputs()[0].name

    def fit_transform(self, X, epochs=50, batch_size=32):
        """Train and encode (requires TensorFlow)."""
        if not HAS_TF or self.autoencoder is None:
            raise RuntimeError("TensorFlow/Keras is required for training.")
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
        """Encode features. Uses ONNX session if Keras encoder is None."""
        X_scaled = self.scaler.transform(X).astype(np.float32)
        if self._onnx_sess is not None:
            return self._onnx_sess.run(None, {self._onnx_inp_name: X_scaled})[0]
        if self.encoder is not None:
            return self.encoder.predict(X_scaled, verbose=0)
        raise RuntimeError(
            "No encoder available. Call load_onnx_encoder() with the ONNX path first."
        )