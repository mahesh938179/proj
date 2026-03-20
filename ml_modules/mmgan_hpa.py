# mmgan_hpa.py
"""
MMGAN-HPA: Multi-Model Generative Adversarial Network Hybrid Prediction Algorithm
Algorithm 2: Combines MM-HPA pre-processing with GAN-HPA
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from feature_extraction import FeatureExtractor
from autoencoders import StackedAutoencoder
from feature_selection import FeatureSelector
from stock_gan import StockGAN, HyperparameterTuner
from mm_hpa import MMHPA


class MMGAN_HPA:
    """
    Complete MMGAN-HPA Implementation.
    
    Architecture (Fig. 4):
    MM-HPA ──┐
             ├──> GAN-HPA ──> MMGAN-HPA ──> Predicted Stock Prices
    GAN ─────┘
    """
    
    def __init__(self, sequence_length=20, n_pca_components=10, 
                 autoencoder_dims=[128, 64, 32], optimize_hyperparams=False):
        self.sequence_length = sequence_length
        self.n_pca_components = n_pca_components
        self.autoencoder_dims = autoencoder_dims
        self.optimize_hyperparams = optimize_hyperparams
        
        self.feature_extractor = FeatureExtractor()
        self.autoencoder = None
        self.feature_selector = None
        self.stock_gan = None
        self.mm_hpa = MMHPA(sequence_length)
        
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
    
    # mmgan_hpa.py (CORRECTED - Key sections only)

    def run(self, df, ticker_name='STOCK', gan_epochs=100, mm_epochs=50,
            bayesian_iterations=5):
        """
        Execute the complete MMGAN-HPA algorithm.
        """
        print(f"\n{'='*60}")
        print(f"  MMGAN-HPA for {ticker_name}")
        print(f"{'='*60}")
        
        # Clean data first
        df = df.copy()
        df = df.dropna()
        
        close_prices = df['Close'].values
        if close_prices.ndim > 1:
            close_prices = close_prices.flatten()
        
        # ==============================================
        # PHASE 1: Feature Extraction
        # ==============================================
        print("\n--- Phase 1: Feature Extraction ---")
        all_features = self.feature_extractor.extract_all_features(df)
        
        # Get aligned close prices
        valid_indices = all_features.index
        close_aligned = df.loc[valid_indices, 'Close'].values
        if close_aligned.ndim > 1:
            close_aligned = close_aligned.flatten()
        
        print(f"  Aligned close prices shape: {close_aligned.shape}")
        print(f"  All features shape: {all_features.shape}")
        
        # ... rest of the code
        
        # ==============================================
        # PHASE 2: Stacked Autoencoders (Step 3 - denoising)
        # ==============================================
        print("\n--- Phase 2: Stacked Autoencoders ---")
        feature_matrix = all_features.values
        
        self.autoencoder = StackedAutoencoder(
            input_dim=feature_matrix.shape[1],
            encoding_dims=self.autoencoder_dims
        )
        encoded_features = self.autoencoder.fit_transform(
            feature_matrix, epochs=30, batch_size=32
        )
        print(f"  Encoded features shape: {encoded_features.shape}")
        
        # Combine original features with encoded features
        combined_features = np.column_stack([feature_matrix, encoded_features])
        
        # ==============================================
        # PHASE 3: MM-HPA Pre-processing (Algorithm 2 Steps 1-8)
        # ==============================================
        print("\n--- Phase 3: MM-HPA Pre-processing ---")
        mm_predictions, mm_features, prices_scaled = self.mm_hpa.fit_predict(
            close_aligned, combined_features, epochs=mm_epochs
        )
        
        # Add MM-HPA predictions as additional features
        mm_pred_padded = np.zeros(len(close_aligned))
        mm_pred_padded[self.sequence_length:self.sequence_length+len(mm_predictions)] = mm_predictions
        
        final_features = np.column_stack([
            combined_features, 
            mm_pred_padded.reshape(-1, 1)
        ])
        
        # ==============================================
        # PHASE 4: Feature Importance & Dimensionality Reduction (Step 5)
        # ==============================================
        print("\n--- Phase 4: Feature Selection (XGBoost + PCA) ---")
        self.feature_selector = FeatureSelector(
            n_pca_components=self.n_pca_components,
            importance_threshold=0.005
        )
        
        selected_features, importances = self.feature_selector.fit_transform(
            final_features, close_aligned
        )
        
        # ==============================================
        # PHASE 5: Prepare data for GAN
        # ==============================================
        print("\n--- Phase 5: Preparing GAN Data ---")
        
        # Scale features and prices
        features_scaled = self.feature_scaler.fit_transform(selected_features)
        prices_for_gan = self.price_scaler.fit_transform(
            close_aligned.reshape(-1, 1)
        ).flatten()
        
        # Create sequences
        X_sequences, y_sequences = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X_sequences.append(features_scaled[i:i + self.sequence_length])
            y_sequences.append(prices_for_gan[i + self.sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Train/test split (80/20)
        split_idx = int(len(X_sequences) * 0.8)
        X_train = X_sequences[:split_idx]
        y_train = y_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_test = y_sequences[split_idx:]
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        n_features = X_train.shape[2]
        
        # ==============================================
        # PHASE 6: Hyperparameter Optimization (Step 10/13)
        # ==============================================
        if self.optimize_hyperparams:
            print("\n--- Phase 6: Bayesian Hyperparameter Optimization ---")
            tuner = HyperparameterTuner(
                self.sequence_length, n_features,
                X_train, y_train,
                X_test[:len(X_test)//2], y_test[:len(y_test)//2]
            )
            best_hyperparams = tuner.optimize(n_iter=bayesian_iterations)
        else:
            best_hyperparams = None  # Use defaults
        
        # ==============================================
        # PHASE 7: Train Stock-GAN (Algorithm 2 Steps 9-15)
        # ==============================================
        print("\n--- Phase 7: Training Stock-GAN ---")
        self.stock_gan = StockGAN(
            sequence_length=self.sequence_length,
            n_features=n_features,
            hyperparams=best_hyperparams
        )
        
        d_losses, g_losses = self.stock_gan.train(
            X_train, y_train, 
            epochs=gan_epochs, 
            batch_size=32
        )
        
        # ==============================================
        # PHASE 8: Generate Predictions (Steps 14, 16-17)
        # ==============================================
        print("\n--- Phase 8: Generating Predictions ---")
        
        # Train predictions
        train_pred_scaled = self.stock_gan.predict(X_train)
        train_predictions = self.price_scaler.inverse_transform(
            train_pred_scaled.reshape(-1, 1)
        ).flatten()
        
        # Test predictions
        test_pred_scaled = self.stock_gan.predict(X_test)
        test_predictions = self.price_scaler.inverse_transform(
            test_pred_scaled.reshape(-1, 1)
        ).flatten()
        
        # Actual values
        train_actual = self.price_scaler.inverse_transform(
            y_train.reshape(-1, 1)
        ).flatten()
        test_actual = self.price_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # All predictions
        all_pred_scaled = self.stock_gan.predict(X_sequences)
        all_predictions = self.price_scaler.inverse_transform(
            all_pred_scaled.reshape(-1, 1)
        ).flatten()
        all_actual = self.price_scaler.inverse_transform(
            y_sequences.reshape(-1, 1)
        ).flatten()
        
        # ==============================================
        # PHASE 9: Compute Metrics (Step 16)
        # Paper Eqs. (16) and (17):
        # MAE = (1/n) * Σ|y - ŷ|
        # MSE = (1/n) * Σ(y - ŷ)²
        # ==============================================
        print("\n--- Phase 9: Computing Metrics ---")
        
        results = self._compute_metrics(
            train_actual, train_predictions,
            test_actual, test_predictions,
            all_actual, all_predictions,
            ticker_name
        )
        
        results['d_losses'] = d_losses
        results['g_losses'] = g_losses
        results['all_actual'] = all_actual
        results['all_predictions'] = all_predictions
        results['train_actual'] = train_actual
        results['train_predictions'] = train_predictions
        results['test_actual'] = test_actual
        results['test_predictions'] = test_predictions
        
        return results
    
    def _compute_metrics(self, train_actual, train_pred, test_actual, test_pred,
                         all_actual, all_pred, ticker_name):
        """
        Compute MAE, MSE, and Correlation.
        Paper Eqs. (16) and (17).
        """
        # Normalize for metric computation (as in the paper's results)
        actual_range = np.max(all_actual) - np.min(all_actual)
        
        # Overall metrics
        mae = np.mean(np.abs(all_actual - all_pred)) / actual_range
        mse = np.mean((all_actual - all_pred) ** 2) / (actual_range ** 2)
        correlation = np.corrcoef(all_actual, all_pred)[0, 1]
        
        # Train metrics
        train_mae = np.mean(np.abs(train_actual - train_pred)) / actual_range
        train_mse = np.mean((train_actual - train_pred) ** 2) / (actual_range ** 2)
        
        # Test metrics
        test_mae = np.mean(np.abs(test_actual - test_pred)) / actual_range
        test_mse = np.mean((test_actual - test_pred) ** 2) / (actual_range ** 2)
        test_corr = np.corrcoef(test_actual, test_pred)[0, 1]
        
        print(f"\n  Results for {ticker_name}:")
        print(f"  {'Metric':<15} {'Train':<15} {'Test':<15} {'Overall':<15}")
        print(f"  {'-'*55}")
        print(f"  {'MAE':<15} {train_mae:<15.8f} {test_mae:<15.8f} {mae:<15.8f}")
        print(f"  {'MSE':<15} {train_mse:<15.8f} {test_mse:<15.8f} {mse:<15.8f}")
        print(f"  {'Correlation':<15} {'N/A':<15} {test_corr:<15.8f} {correlation:<15.8f}")
        
        return {
            'ticker': ticker_name,
            'MAE': mae,
            'MSE': mse,
            'CORRELATION': correlation,
            'train_MAE': train_mae,
            'train_MSE': train_mse,
            'test_MAE': test_mae,
            'test_MSE': test_mse,
            'test_CORRELATION': test_corr,
        }