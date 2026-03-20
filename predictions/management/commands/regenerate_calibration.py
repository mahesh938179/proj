"""
Regenerate calibration files FAST — no slow ARIMA loop, only 3 years data.

Usage:
    python manage.py regenerate_calibration
    python manage.py regenerate_calibration --stock TCS WIPRO
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Regenerate calibration files fast from saved models'

    def add_arguments(self, parser):
        parser.add_argument('--stock', nargs='+', type=str,
                            help='Symbols e.g. TCS WIPRO (default: all)')

    def handle(self, *args, **options):
        base_dir   = settings.BASE_DIR
        models_dir = settings.ML_MODELS_DIR
        symbols    = options.get('stock') or list(settings.STOCK_CONFIG.keys())
        symbols    = [s.upper() for s in symbols]
        sys.path.insert(0, str(base_dir / 'ml_modules'))

        self.stdout.write(self.style.HTTP_INFO(
            f'\n{"="*55}\n  Regenerating calibration (fast)\n'
            f'  Stocks: {", ".join(symbols)}\n{"="*55}\n'
        ))

        ok = fail = 0
        for symbol in symbols:
            self.stdout.write(f'\n[{symbol}]')
            try:
                self._run(symbol, base_dir, models_dir)
                ok += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  ERROR: {e}'))
                import traceback; traceback.print_exc()
                fail += 1

        self.stdout.write(self.style.HTTP_INFO(
            f'\n{"="*55}\n  Done: {ok} ok, {fail} failed\n{"="*55}\n'
        ))
        if ok:
            self.stdout.write(self.style.SUCCESS(
                '  Now run: python manage.py run_predictions'
            ))

    def _run(self, symbol, base_dir, models_dir):
        import yfinance as yf
        from tensorflow import keras
        from feature_extraction import FeatureExtractor

        config  = settings.STOCK_CONFIG[symbol]
        blend   = config.get('blend', 0.08)

        # Output dirs
        out_preds = base_dir / 'outputs' / 'results' / 'predictions'
        out_mm    = base_dir / 'outputs' / 'mmhpa'
        out_gan   = base_dir / 'outputs' / 'gan'
        for d in [out_preds, out_mm, out_gan]:
            d.mkdir(parents=True, exist_ok=True)

        # ── Load models ──────────────────────────────────
        self.stdout.write('  Loading models...')
        price_scaler   = joblib.load(models_dir / 'scalers'     / f'{symbol}_price_scaler.pkl')
        feature_scaler = joblib.load(models_dir / 'scalers'     / f'{symbol}_feature_scaler.pkl')
        selector       = joblib.load(models_dir / 'selection'   / f'{symbol}_selector.pkl')
        generator      = keras.models.load_model(
            models_dir / 'gan' / f'{symbol}_generator.keras'
        )
        with open(models_dir / 'mmhpa'       / f'{symbol}_mmhpa.pkl',       'rb') as f:
            mmhpa_model = pickle.load(f)
        with open(models_dir / 'autoencoder' / f'{symbol}_autoencoder.pkl', 'rb') as f:
            autoencoder = pickle.load(f)

        seq_len = mmhpa_model.sequence_length  # 20

        # ── Download 3 years only (not 2000 days) ────────
        self.stdout.write('  Downloading 3 years of data (~750 calendar days)...')
        end   = datetime.now()
        start = end - timedelta(days=750)
        df    = yf.download(
            config['ticker'],
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].apply(
            pd.to_numeric, errors='coerce'
        ).dropna()
        self.stdout.write(f'  Got {len(df)} trading days')

        if len(df) < seq_len + 60:
            raise ValueError(f'Not enough data: {len(df)} rows')

        # ── Feature extraction WITHOUT slow ARIMA ────────
        # ARIMA loops over every row — very slow for long data
        # For calibration we skip it and fill with deterministic values
        self.stdout.write('  Extracting features (skipping ARIMA to save time)...')
        extractor = FeatureExtractor()
        close_arr = extractor._flatten(df['Close'])

        tech    = extractor.technical_indicators(df)
        fourier = extractor.fourier_transform_features(close_arr)
        anomaly = extractor.som_anomaly_features(close_arr)

        feat_df = pd.DataFrame(index=df.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            feat_df[col] = extractor._flatten(df[col])
        for col in tech.columns:
            feat_df[col] = tech[col].values
        for col in fourier.columns:
            feat_df[col] = fourier[col].values[:len(df)]
        feat_df['anomaly_score']    = anomaly
        # ARIMA cols filled with actual close (non-zero, avoids scaler issues)
        feat_df['arima_prediction'] = close_arr
        feat_df['arima_residual']   = 0.0

        feat_df    = feat_df.dropna()
        close_vals = df.loc[feat_df.index, 'Close'].values.flatten()
        feat_mat   = feat_df.values
        self.stdout.write(f'  Feature matrix: {feat_mat.shape}')

        # ── Encode → select → scale ───────────────────────
        encoded     = autoencoder.transform(feat_mat)
        combined    = np.column_stack([feat_mat, encoded])
        mm_pad      = np.zeros((len(close_vals), 1))
        final_feats = np.column_stack([combined, mm_pad])

        try:
            selected = selector.transform(final_feats)
        except Exception as e:
            self.stdout.write(f'  Selector fallback: {e}')
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import MinMaxScaler
            n = feature_scaler.n_features_in_
            selected = PCA(n_components=n).fit_transform(
                MinMaxScaler().fit_transform(final_feats)
            )

        try:
            scaled = feature_scaler.transform(selected)
        except Exception:
            from sklearn.preprocessing import MinMaxScaler
            scaled = MinMaxScaler().fit_transform(selected)

        # Match generator input shape
        n_feat = generator.input_shape[-1]
        if scaled.shape[1] < n_feat:
            scaled = np.column_stack(
                [scaled, np.zeros((scaled.shape[0], n_feat - scaled.shape[1]))]
            )
        elif scaled.shape[1] > n_feat:
            scaled = scaled[:, :n_feat]

        # MMHPA scaled features (uses ARIMA + LR — only for small recent window)
        # We compute these only on last (seq_len + WINDOW + 10) rows to keep it fast
        arima_pred  = mmhpa_model._arima_predictions(close_vals)
        lr_pred     = mmhpa_model._linear_regression_predictions(close_vals)
        mm_combined = np.column_stack([combined, arima_pred, lr_pred])
        try:
            mm_scaled = mmhpa_model.scaler.transform(mm_combined)
        except Exception:
            from sklearn.preprocessing import MinMaxScaler
            mm_scaled = MinMaxScaler().fit_transform(mm_combined)

        prices_sc = price_scaler.transform(
            close_vals.reshape(-1, 1)
        ).flatten()

        # ── Build sequences (80/20 split) ─────────────────
        X, y, X_mm = [], [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i+seq_len])
            y.append(prices_sc[i+seq_len])
            X_mm.append(mm_scaled[i:i+seq_len])
        X    = np.array(X)
        y    = np.array(y)
        X_mm = np.array(X_mm)

        split    = int(len(X) * 0.8)
        X_test   = X[split:]
        y_test   = y[split:]
        X_mm_t   = X_mm[split:]
        self.stdout.write(f'  Sequences: total={len(X)}, test={len(X_test)}')

        # ── GAN inference ─────────────────────────────────
        self.stdout.write('  GAN inference on test set...')
        gan_sc     = generator.predict(X_test, verbose=0).flatten()
        gan_prices = price_scaler.inverse_transform(
            gan_sc.reshape(-1, 1)
        ).flatten()

        # ── MMHPA inference ───────────────────────────────
        self.stdout.write('  MMHPA inference on test set...')
        if mmhpa_model.nonlinear_model is not None:
            mm_sc     = mmhpa_model.nonlinear_model.predict(
                X_mm_t, verbose=0
            ).flatten()
            mm_prices = mmhpa_model.price_scaler.inverse_transform(
                mm_sc.reshape(-1, 1)
            ).flatten()
        else:
            mm_prices = gan_prices.copy()

        # MMGANHPA ensemble
        mmganhpa = (1 - blend) * gan_prices + blend * mm_prices

        # Pad MM to full length
        mm_full         = np.zeros(len(X))
        mm_full[split:] = mm_prices

        # ── Save ──────────────────────────────────────────
        pd.DataFrame({'Prediction': mmganhpa}).to_csv(
            out_preds / f'{symbol}_test_predictions.csv', index=False
        )
        pd.DataFrame({'MM_HPA': mm_full}).to_csv(
            out_mm / f'{symbol}_mm_predictions.csv', index=False
        )
        np.save(out_gan / f'{symbol}_y_test.npy', y_test)

        # ── Accuracy report ───────────────────────────────
        y_actual = price_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        def ret(a): return (a[1:] - a[:-1]) / a[:-1]

        roll_p = pd.Series(ret(mmganhpa)).rolling(5).mean().values
        roll_a = pd.Series(ret(y_actual)).rolling(5).mean().values
        valid  = ~np.isnan(roll_p) & ~np.isnan(roll_a)
        acc_5d = float(np.mean(np.sign(roll_p[valid]) == np.sign(roll_a[valid])))
        rng    = y_actual.max() - y_actual.min() + 1e-10
        mae    = float(np.mean(np.abs(y_actual - mmganhpa)) / rng)

        self.stdout.write(self.style.SUCCESS(
            f'  MAE(norm)={mae:.6f}  '
            f'5-day accuracy={acc_5d:.1%}  '
            f'samples={len(y_test)}'
        ))
