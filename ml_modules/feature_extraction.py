# predictor/ml_engine/feature_extraction.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def _flatten(self, data):
        """Helper to ensure data is 1D array"""
        if isinstance(data, pd.Series):
            return data.values.flatten()
        elif isinstance(data, pd.DataFrame):
            return data.values.flatten()
        else:
            return np.array(data).flatten()

    def fourier_transform_features(self, close_prices, n_components_list=[3, 6, 9]):
        close_prices = self._flatten(close_prices)
        close_fft = np.fft.fft(close_prices)
        fft_df = pd.DataFrame()

        for n_comp in n_components_list:
            fft_filtered = close_fft.copy()
            fft_filtered[n_comp:-n_comp] = 0
            trend = np.fft.ifft(fft_filtered).real
            fft_df[f'fourier_{n_comp}'] = trend

        frequencies = np.fft.fftfreq(len(close_prices))
        magnitudes = np.abs(close_fft)
        top_indices = np.argsort(magnitudes)[-10:]

        for i, idx in enumerate(top_indices):
            fft_df[f'fft_magnitude_{i}'] = magnitudes[idx]
            fft_df[f'fft_frequency_{i}'] = frequencies[idx]

        return fft_df

    def arima_features(self, close_prices, order=(2, 1, 1), window=60):
        close_prices = self._flatten(close_prices)
        arima_predictions = []
        arima_residuals = []

        for i in range(len(close_prices)):
            if i < window:
                arima_predictions.append(close_prices[i])
                arima_residuals.append(0)
            else:
                try:
                    train_data = close_prices[i - window:i]
                    model = ARIMA(train_data, order=order)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=1)[0]
                    arima_predictions.append(forecast)
                    arima_residuals.append(close_prices[i] - forecast)
                except:
                    arima_predictions.append(close_prices[i])
                    arima_residuals.append(0)

        return np.array(arima_predictions), np.array(arima_residuals)

    def technical_indicators(self, df):
        features = pd.DataFrame(index=df.index)

        close = self._flatten(df['Close'])
        high = self._flatten(df['High'])
        low = self._flatten(df['Low'])
        volume = self._flatten(df['Volume'])
        open_p = self._flatten(df['Open'])

        s_close = pd.Series(close, index=df.index)
        s_volume = pd.Series(volume, index=df.index)

        for period in [7, 14, 21, 50]:
            features[f'SMA_{period}'] = s_close.rolling(window=period).mean().values
            features[f'EMA_{period}'] = s_close.ewm(span=period).mean().values

        delta = s_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        features['RSI'] = (100 - (100 / (1 + rs))).values

        ema12 = s_close.ewm(span=12).mean()
        ema26 = s_close.ewm(span=26).mean()
        features['MACD'] = (ema12 - ema26).values
        features['MACD_signal'] = (ema12 - ema26).ewm(span=9).mean().values

        sma20 = s_close.rolling(window=20).mean()
        std20 = s_close.rolling(window=20).std()
        features['BB_upper'] = (sma20 + 2 * std20).values
        features['BB_lower'] = (sma20 - 2 * std20).values
        features['BB_width'] = ((features['BB_upper'] - features['BB_lower']) / (sma20 + 1e-10)).values

        features['volatility_14'] = s_close.pct_change().rolling(14).std().values
        features['volatility_30'] = s_close.pct_change().rolling(30).std().values

        features['volume_sma_14'] = s_volume.rolling(14).mean().values
        features['volume_ratio'] = volume / (features['volume_sma_14'] + 1e-10)

        features['high_low_ratio'] = high / (low + 1e-10)
        features['close_open_ratio'] = close / (open_p + 1e-10)

        features['daily_return'] = s_close.pct_change().values
        features['log_return'] = np.log(close / (np.roll(close, 1) + 1e-10))
        features.iloc[0, -1] = 0

        return features

    def som_anomaly_features(self, data, som_x=10, som_y=10):
        data = self._flatten(data)
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data.reshape(-1, 1))

        window_size = 5
        som_input = []
        for i in range(len(data_normalized) - window_size + 1):
            som_input.append(data_normalized[i:i + window_size].flatten())
        som_input = np.array(som_input)

        if len(som_input) == 0:
            return np.zeros(len(data))

        try:
            som = MiniSom(som_x, som_y, som_input.shape[1], sigma=1.0, learning_rate=0.5)
            som.train_random(som_input, 100)
            anomaly_scores = np.zeros(len(data))
            for i, x in enumerate(som_input):
                anomaly_scores[i + window_size - 1] = som.quantization_error([x])
            return anomaly_scores
        except:
            return np.zeros(len(data))

    def extract_all_features(self, df):
        df = df.copy().dropna()
        close = self._flatten(df['Close'])

        print("    Extracting technical indicators...")
        tech_features = self.technical_indicators(df)

        print("    Computing Fourier transforms (Step 1)...")
        fourier_features = self.fourier_transform_features(close)

        print("    Computing ARIMA predictions (Step 2)...")
        arima_pred, arima_resid = self.arima_features(close, window=30)

        print("    Computing SOM anomaly scores...")
        anomaly_scores = self.som_anomaly_features(close)

        all_features = pd.DataFrame(index=df.index)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                all_features[col] = self._flatten(df[col])

        for col in tech_features.columns:
            all_features[col] = tech_features[col].values

        for col in fourier_features.columns:
            vals = fourier_features[col].values
            if len(vals) > len(df):
                vals = vals[:len(df)]
            elif len(vals) < len(df):
                vals = np.pad(vals, (0, len(df) - len(vals)), mode='edge')
            all_features[col] = vals

        all_features['arima_prediction'] = arima_pred[:len(df)]
        all_features['arima_residual'] = arima_resid[:len(df)]
        all_features['anomaly_score'] = anomaly_scores[:len(df)]

        all_features = all_features.dropna()
        print(f"    Total features: {all_features.shape[1]}, Rows: {all_features.shape[0]}")
        return all_features