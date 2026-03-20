# predictor/ml_engine/prediction_service.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings
import traceback

warnings.filterwarnings('ignore')
logger = logging.getLogger('predictor')

from .feature_extraction import FeatureExtractor
from .autoencoders import StackedAutoencoder
from .feature_selection import FeatureSelector
from .stock_gan import StockGAN
from .mm_hpa import MMHPA


class StockPredictionService:

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        _base = getattr(settings, 'BASE_DIR', Path(__file__).resolve().parent.parent)
        self.model_dir = str(getattr(settings, 'ML_MODELS_DIR', _base / 'outputs' / 'models'))
        self.data_dir  = str(getattr(settings, 'ML_DATA_DIR',   _base / 'data'))
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def _flatten(self, data):
        if isinstance(data, pd.Series):
            return data.values.flatten()
        elif isinstance(data, pd.DataFrame):
            return data.values.flatten()
        return np.array(data).flatten()

    def _create_phase_dirs(self, ticker_name):
        """Create all phase subdirectories for a ticker"""
        base = os.path.join(self.data_dir, ticker_name)
        phases = [
            'raw',
            'phase1_features',
            'phase2_autoencoder',
            'phase3_mmhpa',
            'phase4_feature_selection',
            'phase5_sequences',
            'phase6_gan_training',
            'phase7_predictions',
            'phase8_metrics',
        ]
        for phase in phases:
            os.makedirs(os.path.join(base, phase), exist_ok=True)
        return base

    def _save_csv(self, data, path, index_label=None):
        """Helper to save data as CSV"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index_label=index_label)
        elif isinstance(data, pd.Series):
            data.to_csv(path, header=True)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                pd.DataFrame({'value': data}).to_csv(path, index=False)
            else:
                pd.DataFrame(data).to_csv(path, index=False)
        logger.info(f"    Saved: {path}")

    def _save_json(self, data, path):
        """Helper to save data as JSON"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"    Saved: {path}")

    # ========================================
    # DATA FETCHING
    # ========================================

    def fetch_stock_data(self, ticker_name, symbol, days=2000):
        logger.info(f"Fetching data for {ticker_name} ({symbol}), days={days}...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if df is None or len(df) == 0:
                logger.error(f"No data returned for {ticker_name}")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            wanted = ['Open', 'High', 'Low', 'Close', 'Volume']
            available = [c for c in wanted if c in df.columns]
            df = df[available]
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            # Save to main data folder (backward compatible)
            csv_path = os.path.join(self.data_dir, f'{ticker_name}.csv')
            df.to_csv(csv_path)

            logger.info(f"Fetched {len(df)} records for {ticker_name}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker_name}: {e}")
            return None

    def load_stock_data(self, ticker_name):
        csv_path = os.path.join(self.data_dir, f'{ticker_name}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df.apply(pd.to_numeric, errors='coerce').dropna()
        return None

    def get_latest_price(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if len(data) > 0:
                close_val = float(data['Close'].iloc[-1])
                open_val = float(data['Open'].iloc[-1])
                return {
                    'price': close_val,
                    'open': open_val,
                    'high': float(data['High'].iloc[-1]),
                    'low': float(data['Low'].iloc[-1]),
                    'volume': int(data['Volume'].iloc[-1]),
                    'change': round(close_val - open_val, 2),
                    'change_pct': round(
                        (close_val - open_val) / (open_val + 1e-10) * 100, 2
                    ),
                }
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None

    # ========================================
    # MODEL TRAINING WITH PHASE DATA SAVING
    # ========================================

    def train_model(self, ticker_name, symbol, callback=None):
        logger.info(f"=== Starting training for {ticker_name} ===")

        # Create all phase directories
        data_base = self._create_phase_dirs(ticker_name)

        def report(progress, message):
            logger.info(f"[{ticker_name}] {progress}% - {message}")
            if callback:
                callback(progress, message)

        try:
            # ==============================================
            # PHASE 0: RAW DATA
            # ==============================================
            report(5, "Fetching stock data...")
            df = self.fetch_stock_data(ticker_name, symbol, days=2000)
            if df is None or len(df) < 200:
                raise ValueError(
                    f"Insufficient data for {ticker_name} "
                    f"(need 200+, got {len(df) if df is not None else 0})"
                )

            # Save raw data
            raw_dir = os.path.join(data_base, 'raw')
            self._save_csv(
                df,
                os.path.join(raw_dir, f'{ticker_name}_raw_data.csv'),
                index_label='Date'
            )
            self._save_json({
                'ticker': ticker_name,
                'symbol': symbol,
                'total_records': len(df),
                'date_range_start': str(df.index[0]),
                'date_range_end': str(df.index[-1]),
                'columns': list(df.columns),
                'fetched_at': datetime.now().isoformat(),
            }, os.path.join(raw_dir, 'data_info.json'))

            # ==============================================
            # PHASE 1: FEATURE EXTRACTION
            # ==============================================
            report(10, "Phase 1: Extracting features...")
            phase1_dir = os.path.join(data_base, 'phase1_features')

            close = self._flatten(df['Close'])

            # Technical indicators
            tech_features = self.feature_extractor.technical_indicators(df)
            self._save_csv(
                tech_features,
                os.path.join(phase1_dir, 'technical_indicators.csv'),
                index_label='Date'
            )

            # Fourier features
            fourier_features = self.feature_extractor.fourier_transform_features(close)
            self._save_csv(
                fourier_features,
                os.path.join(phase1_dir, 'fourier_features.csv')
            )

            # ARIMA features
            arima_pred, arima_resid = self.feature_extractor.arima_features(close, window=30)
            arima_df = pd.DataFrame({
                'arima_prediction': arima_pred,
                'arima_residual': arima_resid,
                'actual_close': close,
            })
            self._save_csv(
                arima_df,
                os.path.join(phase1_dir, 'arima_features.csv')
            )

            # SOM anomaly scores
            anomaly_scores = self.feature_extractor.som_anomaly_features(close)
            self._save_csv(
                pd.DataFrame({
                    'anomaly_score': anomaly_scores,
                    'close_price': close,
                }),
                os.path.join(phase1_dir, 'som_anomaly_scores.csv')
            )

            # Combined features
            all_features = self.feature_extractor.extract_all_features(df)
            self._save_csv(
                all_features,
                os.path.join(phase1_dir, 'all_features_combined.csv'),
                index_label='Date'
            )

            valid_indices = all_features.index
            close_aligned = self._flatten(df.loc[valid_indices, 'Close'])
            feature_matrix = all_features.values

            n_raw_features = feature_matrix.shape[1]

            self._save_json({
                'total_features': n_raw_features,
                'total_rows': feature_matrix.shape[0],
                'feature_names': list(all_features.columns),
                'features_breakdown': {
                    'ohlcv': 5,
                    'technical_indicators': len(tech_features.columns),
                    'fourier_features': len(fourier_features.columns),
                    'arima_features': 2,
                    'som_features': 1,
                },
            }, os.path.join(phase1_dir, 'feature_info.json'))

            # ==============================================
            # PHASE 2: STACKED AUTOENCODERS
            # ==============================================
            report(20, "Phase 2: Training Stacked Autoencoders...")
            phase2_dir = os.path.join(data_base, 'phase2_autoencoder')

            autoencoder = StackedAutoencoder(
                input_dim=feature_matrix.shape[1],
                encoding_dims=[128, 64, 32]
            )
            encoded = autoencoder.fit_transform(
                feature_matrix, epochs=30, batch_size=32
            )

            # Save encoded features
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f'encoded_{i}' for i in range(encoded.shape[1])]
            )
            self._save_csv(
                encoded_df,
                os.path.join(phase2_dir, 'encoded_features.csv')
            )

            combined = np.column_stack([feature_matrix, encoded])
            n_combined_features = combined.shape[1]

            # Save combined
            combined_cols = (
                list(all_features.columns) +
                [f'encoded_{i}' for i in range(encoded.shape[1])]
            )
            combined_df = pd.DataFrame(combined, columns=combined_cols)
            self._save_csv(
                combined_df,
                os.path.join(phase2_dir, 'combined_with_encoded.csv')
            )

            self._save_json({
                'input_dim': feature_matrix.shape[1],
                'encoding_dims': [128, 64, 32],
                'bottleneck_dim': encoded.shape[1],
                'combined_features': n_combined_features,
                'original_features': n_raw_features,
                'encoded_features': encoded.shape[1],
            }, os.path.join(phase2_dir, 'autoencoder_info.json'))

            # ==============================================
            # PHASE 3: MM-HPA PRE-PROCESSING
            # ==============================================
            report(30, "Phase 3: Running MM-HPA pre-processing...")
            phase3_dir = os.path.join(data_base, 'phase3_mmhpa')

            mm_hpa = MMHPA(sequence_length=20)
            mm_pred, mm_features, prices_scaled = mm_hpa.fit_predict(
                close_aligned, combined, epochs=30
            )

            # Save MM-HPA predictions
            self._save_csv(
                pd.DataFrame({
                    'mmhpa_prediction': mm_pred,
                }),
                os.path.join(phase3_dir, 'mmhpa_predictions.csv')
            )

            # Save MM-HPA linear features
            self._save_csv(
                pd.DataFrame(mm_features),
                os.path.join(phase3_dir, 'mmhpa_scaled_features.csv')
            )

            # Create padded MM-HPA column
            mm_padded = np.zeros(len(close_aligned))
            end_idx = min(20 + len(mm_pred), len(close_aligned))
            mm_padded[20:end_idx] = mm_pred[:end_idx - 20]

            final_features = np.column_stack([
                combined, mm_padded.reshape(-1, 1)
            ])
            n_final_features = final_features.shape[1]

            # Save final features
            final_cols = combined_cols + ['mmhpa_prediction']
            final_df = pd.DataFrame(final_features, columns=final_cols)
            self._save_csv(
                final_df,
                os.path.join(phase3_dir, 'final_features_with_mmhpa.csv')
            )

            self._save_json({
                'sequence_length': 20,
                'mm_predictions_count': len(mm_pred),
                'final_features_shape': list(final_features.shape),
                'features_added': ['mmhpa_prediction'],
                'total_features_now': n_final_features,
            }, os.path.join(phase3_dir, 'mmhpa_info.json'))

            # ==============================================
            # PHASE 4: FEATURE SELECTION (XGBoost + PCA)
            # ==============================================
            report(40, "Phase 4: Feature selection (XGBoost + PCA)...")
            phase4_dir = os.path.join(data_base, 'phase4_feature_selection')

            selector = FeatureSelector(
                n_pca_components=10, importance_threshold=0.005
            )
            selected, importances = selector.fit_transform(
                final_features, close_aligned
            )
            n_selected = selected.shape[1]

            # Save XGBoost importances
            importance_df = pd.DataFrame({
                'feature_index': range(len(importances)),
                'feature_name': final_cols[:len(importances)] if len(final_cols) >= len(importances) else [f'feature_{i}' for i in range(len(importances))],
                'importance': importances,
                'selected': [i in selector.selected_features for i in range(len(importances))],
            }).sort_values('importance', ascending=False)
            self._save_csv(
                importance_df,
                os.path.join(phase4_dir, 'xgboost_importances.csv')
            )

            # Save selected features (before PCA)
            selected_feature_names = [
                final_cols[i] if i < len(final_cols) else f'feature_{i}'
                for i in selector.selected_features
            ]
            selected_before_pca = final_features[:, selector.selected_features]
            self._save_csv(
                pd.DataFrame(selected_before_pca, columns=selected_feature_names),
                os.path.join(phase4_dir, 'selected_features_before_pca.csv')
            )

            # Save PCA reduced features
            pca_cols = [f'PC_{i+1}' for i in range(selected.shape[1])]
            self._save_csv(
                pd.DataFrame(selected, columns=pca_cols),
                os.path.join(phase4_dir, 'pca_reduced_features.csv')
            )

            self._save_json({
                'original_features': n_final_features,
                'xgboost_selected': len(selector.selected_features),
                'selected_indices': selector.selected_features.tolist(),
                'selected_feature_names': selected_feature_names,
                'pca_components': n_selected,
                'pca_explained_variance': selector.pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': float(
                    np.sum(selector.pca.explained_variance_ratio_)
                ),
                'importance_threshold': 0.005,
            }, os.path.join(phase4_dir, 'selection_info.json'))

            # ==============================================
            # PHASE 5: SEQUENCES & SCALING
            # ==============================================
            report(50, "Phase 5: Preparing training sequences...")
            phase5_dir = os.path.join(data_base, 'phase5_sequences')

            feature_scaler = MinMaxScaler()
            price_scaler = MinMaxScaler()

            features_scaled = feature_scaler.fit_transform(selected)
            prices_scaled_final = price_scaler.fit_transform(
                close_aligned.reshape(-1, 1)
            ).flatten()

            # Save scaled data
            self._save_csv(
                pd.DataFrame(features_scaled, columns=pca_cols),
                os.path.join(phase5_dir, 'scaled_features.csv')
            )
            self._save_csv(
                pd.DataFrame({
                    'scaled_price': prices_scaled_final,
                    'original_price': close_aligned,
                }),
                os.path.join(phase5_dir, 'scaled_prices.csv')
            )

            # Create sequences
            seq_len = 20
            X, y = [], []
            for i in range(len(features_scaled) - seq_len):
                X.append(features_scaled[i:i + seq_len])
                y.append(prices_scaled_final[i + seq_len])
            X, y = np.array(X), np.array(y)

            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            n_features = X_train.shape[2]

            # Save split info
            self._save_json({
                'sequence_length': seq_len,
                'total_sequences': len(X),
                'train_sequences': len(X_train),
                'test_sequences': len(X_test),
                'split_ratio': 0.8,
                'split_index': split,
                'n_features_per_step': n_features,
                'X_train_shape': list(X_train.shape),
                'X_test_shape': list(X_test.shape),
                'y_train_shape': list(y_train.shape),
                'y_test_shape': list(y_test.shape),
            }, os.path.join(phase5_dir, 'sequence_info.json'))

            # Save train/test target values
            self._save_csv(
                pd.DataFrame({
                    'y_train_scaled': y_train,
                    'y_train_actual': price_scaler.inverse_transform(
                        y_train.reshape(-1, 1)
                    ).flatten(),
                }),
                os.path.join(phase5_dir, 'train_targets.csv')
            )
            self._save_csv(
                pd.DataFrame({
                    'y_test_scaled': y_test,
                    'y_test_actual': price_scaler.inverse_transform(
                        y_test.reshape(-1, 1)
                    ).flatten(),
                }),
                os.path.join(phase5_dir, 'test_targets.csv')
            )

            # ==============================================
            # PHASE 6: STOCK-GAN TRAINING
            # ==============================================
            report(55, "Phase 6: Training Stock-GAN (LSTM + CNN)...")
            phase6_dir = os.path.join(data_base, 'phase6_gan_training')

            gan = StockGAN(
                sequence_length=seq_len, n_features=n_features
            )
            d_losses, g_losses = gan.train(
                X_train, y_train, epochs=100, batch_size=32
            )

            # Save training losses
            self._save_csv(
                pd.DataFrame({
                    'epoch': range(1, len(d_losses) + 1),
                    'discriminator_loss': d_losses,
                    'generator_loss': g_losses,
                }),
                os.path.join(phase6_dir, 'training_losses.csv')
            )
            self._save_csv(
                pd.DataFrame({
                    'epoch': range(1, len(d_losses) + 1),
                    'loss': d_losses,
                }),
                os.path.join(phase6_dir, 'discriminator_losses.csv')
            )
            self._save_csv(
                pd.DataFrame({
                    'epoch': range(1, len(g_losses) + 1),
                    'loss': g_losses,
                }),
                os.path.join(phase6_dir, 'generator_losses.csv')
            )

            self._save_json({
                'generator': 'LSTM (3 layers)',
                'discriminator': 'CNN (2 Conv + 2 FC)',
                'epochs': 100,
                'batch_size': 32,
                'sequence_length': seq_len,
                'n_features': n_features,
                'hyperparameters': gan.hyperparams,
                'final_d_loss': float(d_losses[-1]),
                'final_g_loss': float(g_losses[-1]),
                'min_d_loss': float(min(d_losses)),
                'min_g_loss': float(min(g_losses)),
            }, os.path.join(phase6_dir, 'training_config.json'))

            # ==============================================
            # PHASE 7: PREDICTIONS
            # ==============================================
            report(80, "Phase 7: Generating predictions...")
            phase7_dir = os.path.join(data_base, 'phase7_predictions')

            all_pred = price_scaler.inverse_transform(
                gan.predict(X).reshape(-1, 1)
            ).flatten()
            all_actual = price_scaler.inverse_transform(
                y.reshape(-1, 1)
            ).flatten()

            train_pred = price_scaler.inverse_transform(
                gan.predict(X_train).reshape(-1, 1)
            ).flatten()
            train_actual = price_scaler.inverse_transform(
                y_train.reshape(-1, 1)
            ).flatten()

            test_pred = price_scaler.inverse_transform(
                gan.predict(X_test).reshape(-1, 1)
            ).flatten()
            test_actual = price_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            # Save all predictions
            self._save_csv(
                pd.DataFrame({
                    'actual_price': all_actual,
                    'predicted_price': all_pred,
                    'error': all_actual - all_pred,
                    'abs_error': np.abs(all_actual - all_pred),
                    'pct_error': np.abs(all_actual - all_pred) / (all_actual + 1e-10) * 100,
                }),
                os.path.join(phase7_dir, 'all_predictions.csv')
            )

            # Save train predictions
            self._save_csv(
                pd.DataFrame({
                    'actual_price': train_actual,
                    'predicted_price': train_pred,
                    'error': train_actual - train_pred,
                    'abs_error': np.abs(train_actual - train_pred),
                }),
                os.path.join(phase7_dir, 'train_predictions.csv')
            )

            # Save test predictions
            self._save_csv(
                pd.DataFrame({
                    'actual_price': test_actual,
                    'predicted_price': test_pred,
                    'error': test_actual - test_pred,
                    'abs_error': np.abs(test_actual - test_pred),
                }),
                os.path.join(phase7_dir, 'test_predictions.csv')
            )

            # Combined actual vs predicted
            self._save_csv(
                pd.DataFrame({
                    'index': range(len(all_actual)),
                    'actual': all_actual,
                    'predicted': all_pred,
                    'split': ['train'] * len(train_actual) + ['test'] * len(test_actual),
                }),
                os.path.join(phase7_dir, 'prediction_vs_actual.csv')
            )

            # ==============================================
            # PHASE 8: METRICS
            # ==============================================
            report(85, "Phase 8: Computing metrics...")
            phase8_dir = os.path.join(data_base, 'phase8_metrics')

            rng = np.max(all_actual) - np.min(all_actual)
            if rng == 0:
                rng = 1.0

            metrics = {
                'mae': float(np.mean(np.abs(all_actual - all_pred)) / rng),
                'mse': float(np.mean((all_actual - all_pred) ** 2) / (rng ** 2)),
                'correlation': float(np.corrcoef(all_actual, all_pred)[0, 1]),
                'train_mae': float(np.mean(np.abs(train_actual - train_pred)) / rng),
                'train_mse': float(np.mean((train_actual - train_pred) ** 2) / (rng ** 2)),
                'test_mae': float(np.mean(np.abs(test_actual - test_pred)) / rng),
                'test_mse': float(np.mean((test_actual - test_pred) ** 2) / (rng ** 2)),
                'test_correlation': float(np.corrcoef(test_actual, test_pred)[0, 1]),
            }

            # Additional metrics
            rmse = float(np.sqrt(np.mean((all_actual - all_pred) ** 2)))
            mape = float(np.mean(np.abs(all_actual - all_pred) / (all_actual + 1e-10)) * 100)
            r_squared = float(1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - np.mean(all_actual)) ** 2))

            extended_metrics = {
                **metrics,
                'rmse': rmse,
                'mape': mape,
                'r_squared': r_squared,
                'price_range': float(rng),
                'mean_actual': float(np.mean(all_actual)),
                'mean_predicted': float(np.mean(all_pred)),
                'std_actual': float(np.std(all_actual)),
                'std_predicted': float(np.std(all_pred)),
                'min_actual': float(np.min(all_actual)),
                'max_actual': float(np.max(all_actual)),
                'min_predicted': float(np.min(all_pred)),
                'max_predicted': float(np.max(all_pred)),
                'total_data_points': len(df),
                'training_samples': len(X_train),
                'testing_samples': len(X_test),
                'computed_at': datetime.now().isoformat(),
            }

            self._save_json(
                extended_metrics,
                os.path.join(phase8_dir, 'performance_metrics.json')
            )

            # Metrics summary CSV
            metrics_summary = pd.DataFrame([
                {'metric': 'MAE (normalized)', 'train': metrics['train_mae'], 'test': metrics['test_mae'], 'overall': metrics['mae']},
                {'metric': 'MSE (normalized)', 'train': metrics['train_mse'], 'test': metrics['test_mse'], 'overall': metrics['mse']},
                {'metric': 'Correlation', 'train': 'N/A', 'test': metrics['test_correlation'], 'overall': metrics['correlation']},
                {'metric': 'RMSE (actual)', 'train': '', 'test': '', 'overall': rmse},
                {'metric': 'MAPE (%)', 'train': '', 'test': '', 'overall': mape},
                {'metric': 'R-Squared', 'train': '', 'test': '', 'overall': r_squared},
            ])
            self._save_csv(
                metrics_summary,
                os.path.join(phase8_dir, 'metrics_summary.csv')
            )

            # Per-sample error analysis
            self._save_csv(
                pd.DataFrame({
                    'sample_index': range(len(test_actual)),
                    'actual': test_actual,
                    'predicted': test_pred,
                    'error': test_actual - test_pred,
                    'abs_error': np.abs(test_actual - test_pred),
                    'pct_error': np.abs(test_actual - test_pred) / (test_actual + 1e-10) * 100,
                }),
                os.path.join(phase8_dir, 'test_error_analysis.csv')
            )

            # ==============================================
            # SAVE MODEL FILES
            # ==============================================
            report(90, "Saving model and pipeline...")
            # Save in engine.py-compatible paths:
            # outputs/models/gan/{symbol}_generator.keras
            # outputs/models/autoencoder/{symbol}_autoencoder.pkl
            # outputs/models/selection/{symbol}_selector.pkl
            # outputs/models/scalers/{symbol}_feature_scaler.pkl
            # outputs/models/scalers/{symbol}_price_scaler.pkl
            # outputs/models/mmhpa/{symbol}_mmhpa.pkl  (placeholder)

            gan_dir       = os.path.join(self.model_dir, 'gan')
            ae_dir        = os.path.join(self.model_dir, 'autoencoder')
            sel_dir       = os.path.join(self.model_dir, 'selection')
            scaler_dir    = os.path.join(self.model_dir, 'scalers')
            mmhpa_dir     = os.path.join(self.model_dir, 'mmhpa')
            for d in [gan_dir, ae_dir, sel_dir, scaler_dir, mmhpa_dir]:
                os.makedirs(d, exist_ok=True)

            # Save generator as .keras (engine.py compatible)
            gan.generator.save(os.path.join(gan_dir, f'{ticker_name}_generator.keras'))
            gan.discriminator.save(os.path.join(gan_dir, f'{ticker_name}_discriminator.keras'))

            # Save autoencoder pkl
            import pickle as _pk
            with open(os.path.join(ae_dir, f'{ticker_name}_autoencoder.pkl'), 'wb') as f:
                _pk.dump(autoencoder, f)

            # Save selector pkl
            import joblib as _jl
            _jl.dump(selector, os.path.join(sel_dir, f'{ticker_name}_selector.pkl'))

            # Save scalers
            _jl.dump(feature_scaler, os.path.join(scaler_dir, f'{ticker_name}_feature_scaler.pkl'))
            _jl.dump(price_scaler,   os.path.join(scaler_dir, f'{ticker_name}_price_scaler.pkl'))

            # Save MMHPA model pkl (mm_hpa object)
            with open(os.path.join(mmhpa_dir, f'{ticker_name}_mmhpa.pkl'), 'wb') as f:
                _pk.dump(mm_hpa, f)

            # Also save config in a per-ticker folder for is_model_trained() check
            model_path = os.path.join(self.model_dir, ticker_name)
            os.makedirs(model_path, exist_ok=True)

            # Config
            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump({
                    'sequence_length': seq_len,
                    'n_features': n_features,
                    'n_raw_features': n_raw_features,
                    'n_combined_features': n_combined_features,
                    'n_final_features': n_final_features,
                    'n_selected_features': n_selected,
                    'selected_indices': selector.selected_features.tolist(),
                    'trained_at': datetime.now().isoformat(),
                    'data_points': len(df),
                }, f)

            # Save a master summary in the data folder
            self._save_json({
                'ticker': ticker_name,
                'symbol': symbol,
                'trained_at': datetime.now().isoformat(),
                'phases': {
                    'phase0_raw': {
                        'records': len(df),
                        'date_range': f"{df.index[0]} to {df.index[-1]}",
                    },
                    'phase1_features': {
                        'raw_features': n_raw_features,
                        'rows_after_cleanup': feature_matrix.shape[0],
                    },
                    'phase2_autoencoder': {
                        'encoded_dim': encoded.shape[1],
                        'combined_features': n_combined_features,
                    },
                    'phase3_mmhpa': {
                        'mmhpa_predictions': len(mm_pred),
                        'final_features': n_final_features,
                    },
                    'phase4_selection': {
                        'xgboost_selected': len(selector.selected_features),
                        'pca_components': n_selected,
                    },
                    'phase5_sequences': {
                        'total': len(X),
                        'train': len(X_train),
                        'test': len(X_test),
                    },
                    'phase6_training': {
                        'epochs': 100,
                        'final_d_loss': float(d_losses[-1]),
                        'final_g_loss': float(g_losses[-1]),
                    },
                    'phase7_predictions': {
                        'train_samples': len(train_pred),
                        'test_samples': len(test_pred),
                    },
                    'phase8_metrics': metrics,
                },
            }, os.path.join(data_base, 'training_summary.json'))

            report(100, "Training complete! All phase data saved.")

            return {
                'metrics': metrics,
                'all_actual': all_actual.tolist(),
                'all_predictions': all_pred.tolist(),
                'test_actual': test_actual.tolist(),
                'test_predictions': test_pred.tolist(),
                'd_losses': [float(l) for l in d_losses],
                'g_losses': [float(l) for l in g_losses],
                'data_points': len(df),
            }

        except Exception as e:
            logger.error(f"Training failed for {ticker_name}: {e}")
            traceback.print_exc()
            raise

    # ========================================
    # PREDICTION (ALL FIXES)
    # ========================================

    def predict_next_day(self, ticker_name, symbol):
        from tensorflow.keras.models import model_from_json

        model_path = os.path.join(self.model_dir, ticker_name)

        weights_path = os.path.join(model_path, 'generator.weights.h5')
        arch_path = os.path.join(model_path, 'generator_architecture.json')
        config_path = os.path.join(model_path, 'config.json')
        pipeline_path = os.path.join(model_path, 'pipeline.pkl')
        encoder_weights_path = os.path.join(model_path, 'encoder.weights.h5')
        encoder_arch_path = os.path.join(model_path, 'encoder_architecture.json')
        old_h5_path = os.path.join(model_path, 'generator.h5')

        has_new = os.path.exists(weights_path) and os.path.exists(arch_path)
        has_old = os.path.exists(old_h5_path)

        if not has_new and not has_old:
            raise FileNotFoundError(f"No model for {ticker_name}. Train first.")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config missing for {ticker_name}. Retrain.")

        with open(config_path, 'r') as f:
            config = json.load(f)

        seq_len = config['sequence_length']
        n_features = config['n_features']
        n_final_features = config.get('n_final_features', None)

        # Load generator
        if has_new:
            with open(arch_path, 'r') as f:
                generator = model_from_json(f.read())
            generator.load_weights(weights_path)
            generator.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        else:
            generator = tf.keras.models.load_model(old_h5_path, compile=False)
            generator.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        # Load pipeline
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            raise FileNotFoundError(f"Pipeline missing for {ticker_name}. Retrain.")

        feature_scaler = pipeline['feature_scaler']
        price_scaler = pipeline['price_scaler']
        feature_selector = pipeline['feature_selector']

        # Fetch data
        df = self.fetch_stock_data(ticker_name, symbol, days=2000)
        if df is None:
            df = self.load_stock_data(ticker_name)
        if df is None:
            raise ValueError(f"No data for {ticker_name}")

        # Features
        all_features = self.feature_extractor.extract_all_features(df)
        feature_matrix = all_features.values

        # Autoencoder
        has_encoder = (
            os.path.exists(encoder_weights_path) and
            os.path.exists(encoder_arch_path) and
            'autoencoder_scaler' in pipeline
        )

        if has_encoder:
            try:
                with open(encoder_arch_path, 'r') as f:
                    encoder = model_from_json(f.read())
                encoder.load_weights(encoder_weights_path)

                ae_scaler = pipeline['autoencoder_scaler']
                ae_dim = pipeline.get('autoencoder_input_dim', feature_matrix.shape[1])

                if feature_matrix.shape[1] == ae_dim:
                    feat_ae = feature_matrix
                elif feature_matrix.shape[1] < ae_dim:
                    feat_ae = np.zeros((feature_matrix.shape[0], ae_dim))
                    feat_ae[:, :feature_matrix.shape[1]] = feature_matrix
                else:
                    feat_ae = feature_matrix[:, :ae_dim]

                encoded = encoder.predict(ae_scaler.transform(feat_ae), verbose=0)
                combined = np.column_stack([feature_matrix, encoded])
                feature_matrix = np.column_stack([combined, np.zeros((combined.shape[0], 1))])
            except Exception as e:
                logger.warning(f"Encoder failed: {e}")
                if n_final_features and feature_matrix.shape[1] < n_final_features:
                    pad = np.zeros((feature_matrix.shape[0], n_final_features - feature_matrix.shape[1]))
                    feature_matrix = np.column_stack([feature_matrix, pad])
        else:
            if n_final_features and feature_matrix.shape[1] < n_final_features:
                pad = np.zeros((feature_matrix.shape[0], n_final_features - feature_matrix.shape[1]))
                feature_matrix = np.column_stack([feature_matrix, pad])

        # Feature selection
        try:
            selected = feature_selector.transform(feature_matrix)
        except Exception as e:
            logger.warning(f"Selection failed: {e}. PCA fallback.")
            from sklearn.decomposition import PCA
            n_pca = min(n_features, feature_matrix.shape[1], feature_matrix.shape[0])
            ts = MinMaxScaler()
            selected = PCA(n_components=n_pca).fit_transform(ts.fit_transform(feature_matrix))

        # Scale
        try:
            if selected.shape[1] == feature_scaler.n_features_in_:
                scaled = feature_scaler.transform(selected)
            else:
                scaled = MinMaxScaler().fit_transform(selected)
        except:
            scaled = MinMaxScaler().fit_transform(selected)

        # Fix dimensions
        if scaled.shape[1] != n_features:
            if scaled.shape[1] < n_features:
                scaled = np.column_stack([scaled, np.zeros((scaled.shape[0], n_features - scaled.shape[1]))])
            else:
                scaled = scaled[:, :n_features]

        if len(scaled) < seq_len:
            raise ValueError(f"Need {seq_len} steps, got {len(scaled)}.")

        # Predict
        last_seq = scaled[-seq_len:].reshape(1, seq_len, n_features)
        pred_scaled = generator.predict(last_seq, verbose=0).flatten()
        predicted_price = price_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

        current_price = float(df['Close'].iloc[-1])
        change = predicted_price - current_price

        # Save prediction to data folder
        pred_dir = os.path.join(self.data_dir, ticker_name, 'predictions_log')
        os.makedirs(pred_dir, exist_ok=True)

        pred_log_path = os.path.join(pred_dir, 'prediction_history.csv')
        new_pred = pd.DataFrame([{
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'current_price': current_price,
            'predicted_price': float(predicted_price),
            'change': float(change),
            'change_pct': float(change / (current_price + 1e-10) * 100),
            'direction': 'UP' if change > 0 else 'DOWN',
            'generated_at': datetime.now().isoformat(),
        }])

        if os.path.exists(pred_log_path):
            existing = pd.read_csv(pred_log_path)
            combined_log = pd.concat([existing, new_pred], ignore_index=True)
        else:
            combined_log = new_pred
        combined_log.to_csv(pred_log_path, index=False)

        return {
            'ticker': ticker_name,
            'current_price': round(current_price, 2),
            'predicted_price': round(float(predicted_price), 2),
            'change': round(float(change), 2),
            'change_pct': round(float(change / (current_price + 1e-10) * 100), 2),
            'direction': 'UP' if change > 0 else 'DOWN',
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'last_updated': datetime.now().isoformat(),
        }

    def is_model_trained(self, ticker_name):
        # Check new format: outputs/models/gan/{ticker}_generator.keras
        gan_dir = os.path.join(self.model_dir, 'gan')
        new_format = os.path.exists(
            os.path.join(gan_dir, f'{ticker_name}_generator.keras')
        )
        # Fallback: old per-ticker folder format
        p = os.path.join(self.model_dir, ticker_name)
        old_format = (
            os.path.exists(os.path.join(p, 'generator.weights.h5')) and
            os.path.exists(os.path.join(p, 'generator_architecture.json')) and
            os.path.exists(os.path.join(p, 'config.json'))
        )
        legacy_format = os.path.exists(os.path.join(p, 'generator.h5'))
        return new_format or old_format or legacy_format

    def get_model_info(self, ticker_name):
        config_path = os.path.join(self.model_dir, ticker_name, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None


prediction_service = StockPredictionService()