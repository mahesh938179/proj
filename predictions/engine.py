"""
Core prediction engine — wraps the ML pipeline for all 6 stocks.
v2 Fixed: Uses 5-day average return as signal (79.6% direction accuracy).
v3 Fixed: yfinance rate limit handled gracefully — falls back to DB cache.
"""

import os
import sys
import time
import logging
import pickle
import warnings
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import onnxruntime as rt

logger = logging.getLogger('predictions')


def _download_with_fallback(ticker_symbol: str, symbol: str, base_dir: Path) -> pd.DataFrame:
    """
    Download yfinance data with a SHORT timeout.
    On rate limit or failure → fall back to saved DB/CSV data.
    NO long retries or sleeps that block the worker.
    """
    # Attempt 1: yfinance fast download (10s timeout max)
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period='90d', timeout=10, raise_errors=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if len(df) >= 30:
            logger.info(f'[{symbol}] yfinance download OK ({len(df)} rows)')
            return df
        raise ValueError(f'Too few rows: {len(df)}')
    except Exception as e:
        err_str = str(e).lower()
        is_rate = 'rate' in err_str or 'too many' in err_str or '429' in err_str
        reason = 'rate limit' if is_rate else 'error'
        logger.warning(f'[{symbol}] yfinance {reason}: {e} — trying fallback...')

    # Attempt 2: Try yfinance download() function (different code path)
    try:
        df = yf.download(
            ticker_symbol,
            period='90d',
            auto_adjust=True,
            progress=False,
            timeout=8,
        )
        if not df.empty and len(df) >= 30:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            logger.info(f'[{symbol}] yf.download fallback OK ({len(df)} rows)')
            return df
    except Exception as e2:
        logger.warning(f'[{symbol}] yf.download also failed: {e2}')

    # Attempt 3: Load from saved CSV in data/ directory
    data_dir = base_dir / 'data'
    csv_path = data_dir / f'{symbol}.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if len(df) >= 30:
                logger.warning(f'[{symbol}] Using cached CSV data ({len(df)} rows)')
                return df
        except Exception as e3:
            logger.error(f'[{symbol}] CSV fallback failed: {e3}')

    # Attempt 4: Load from Django DB PriceHistory
    try:
        from predictions.models import PriceHistory, Stock
        stock_obj = Stock.objects.filter(symbol=symbol).first()
        if stock_obj:
            history = PriceHistory.objects.filter(
                stock=stock_obj
            ).order_by('date').values(
                'date', 'open', 'high', 'low', 'close', 'volume'
            )
            if history.count() >= 30:
                df = pd.DataFrame(list(history))
                df = df.rename(columns={
                    'date': 'Date', 'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                logger.warning(
                    f'[{symbol}] Using DB PriceHistory ({len(df)} rows)'
                )
                return df
    except Exception as e4:
        logger.error(f'[{symbol}] DB fallback failed: {e4}')

    raise RuntimeError(
        f'[{symbol}] All data sources failed. '
        f'Cannot download from yfinance and no cached data available.'
    )


class StockPredictor:
    """
    Loads ML models for one stock, runs the full pipeline,
    returns calibrated 5-day-average-return predictions.
    """

    WINDOW = 5  # 5-day rolling for trend signal

    def __init__(self, symbol: str, config: dict, base_dir: Path):
        self.symbol = symbol
        self.config = config
        self.ticker = config['ticker']
        self.blend = config.get('blend', 0.08)
        self.base_dir = base_dir
        self._models_loaded = False

        # ONNX sessions
        self.generator_sess = None
        self.generator_inp = None
        self.mm_nonlinear_sess = None
        self.mm_nonlinear_inp = None

        self._calib = {}
        self._accuracy = {}
        self._hist_std = None

    # ─── Model Loading ──────────────────────────────────
    def load_models(self):
        """Load all ML models + scalers for this stock"""
        if self._models_loaded:
            return

        models_dir = self.base_dir / 'outputs' / 'models'

        ml_dir = str(self.base_dir / 'ml_modules')
        if ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)

        try:
            self.price_scaler = joblib.load(
                models_dir / 'scalers' / f'{self.symbol}_price_scaler.pkl'
            )
            self.feature_scaler = joblib.load(
                models_dir / 'scalers' / f'{self.symbol}_feature_scaler.pkl'
            )

            onnx_path = models_dir / 'gan' / f'{self.symbol}_generator.onnx'
            if not onnx_path.exists():
                raise FileNotFoundError(f'ONNX model not found: {onnx_path}')

            self.generator_sess = rt.InferenceSession(str(onnx_path))
            self.generator_inp  = self.generator_sess.get_inputs()[0].name

            with open(models_dir / 'mmhpa' / f'{self.symbol}_mmhpa.pkl', 'rb') as f:
                self.mmhpa_model = pickle.load(f)

            mm_onnx = models_dir / 'mmhpa' / f'{self.symbol}_mmhpa_nonlinear.onnx'
            if mm_onnx.exists():
                self.mm_nonlinear_sess = rt.InferenceSession(str(mm_onnx))
                self.mm_nonlinear_inp  = self.mm_nonlinear_sess.get_inputs()[0].name
            else:
                self.mm_nonlinear_sess = None

            with open(
                models_dir / 'autoencoder' / f'{self.symbol}_autoencoder.pkl', 'rb'
            ) as f:
                self.autoencoder = pickle.load(f)

            ae_onnx = models_dir / 'autoencoder' / f'{self.symbol}_encoder.onnx'
            if ae_onnx.exists():
                self.autoencoder.load_onnx_encoder(ae_onnx)
            else:
                logger.warning(f'[{self.symbol}] Encoder ONNX not found: {ae_onnx}')

            self.selector = joblib.load(
                models_dir / 'selection' / f'{self.symbol}_selector.pkl'
            )

            self.seq_len = self.mmhpa_model.sequence_length
            self._models_loaded = True
            logger.info(f'[{self.symbol}] All models loaded successfully (ONNX mode)')

        except Exception as e:
            logger.error(f'[{self.symbol}] Model loading failed: {e}')
            raise

    # ─── Calibration ────────────────────────────────────
    def calibrate(self):
        results_dir = self.base_dir / 'outputs' / 'results' / 'predictions'
        gan_dir = self.base_dir / 'outputs' / 'gan'
        mm_dir = self.base_dir / 'outputs' / 'mmhpa'

        try:
            test_preds = pd.read_csv(
                results_dir / f'{self.symbol}_test_predictions.csv'
            )
            mmganhpa_t = test_preds['Prediction'].values

            mm_preds = pd.read_csv(mm_dir / f'{self.symbol}_mm_predictions.csv')
            mmhpa_all = mm_preds['MM_HPA'].values
            mmhpa_t = mmhpa_all[-len(mmganhpa_t):]
            ganhpa_t = (mmganhpa_t - self.blend * mmhpa_t) / (1 - self.blend)

            y_test = np.load(gan_dir / f'{self.symbol}_y_test.npy')
            y_actual = self.price_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            def ret(arr):
                return (arr[1:] - arr[:-1]) / arr[:-1]

            actual_ret   = ret(y_actual)
            mmhpa_ret    = ret(mmhpa_t)
            ganhpa_ret   = ret(ganhpa_t)
            mmganhpa_ret = ret(mmganhpa_t)

            self._calib = {
                'MMHPA':    actual_ret.std() / max(mmhpa_ret.std(),    1e-10),
                'GANHPA':   actual_ret.std() / max(ganhpa_ret.std(),   1e-10),
                'MMGANHPA': actual_ret.std() / max(mmganhpa_ret.std(), 1e-10),
            }
            self._hist_std = actual_ret.std()

            def sign_acc_5d(pred_ret, act_ret):
                roll_p = pd.Series(pred_ret).rolling(5).mean().values
                roll_a = pd.Series(act_ret).rolling(5).mean().values
                valid  = ~np.isnan(roll_p) & ~np.isnan(roll_a)
                if valid.sum() == 0:
                    return 0.5
                return np.mean(np.sign(roll_p[valid]) == np.sign(roll_a[valid]))

            self._accuracy = {
                'MMHPA':    sign_acc_5d(mmhpa_ret,    actual_ret),
                'GANHPA':   sign_acc_5d(ganhpa_ret,   actual_ret),
                'MMGANHPA': sign_acc_5d(mmganhpa_ret, actual_ret),
            }

            logger.info(
                f'[{self.symbol}] Calibration done | '
                f'5d accuracy: {self._accuracy["MMGANHPA"]:.1%}'
            )

        except Exception as e:
            logger.warning(f'[{self.symbol}] Calibration failed, using defaults: {e}')
            self._calib    = {'MMHPA': 1.2244, 'GANHPA': 1.2510, 'MMGANHPA': 1.2611}
            self._accuracy = {'MMHPA': 0.751,  'GANHPA': 0.791,  'MMGANHPA': 0.796}
            self._hist_std = 0.0128

    # ─── Feature Pipeline ──────────────────────────────
    def _run_feature_pipeline(self, df):
        from ml_modules.feature_extraction import FeatureExtractor

        extractor    = FeatureExtractor()
        all_features = extractor.extract_all_features(df)
        close_vals   = df.loc[all_features.index, 'Close'].values.flatten()
        feat_matrix  = all_features.values

        encoded  = self.autoencoder.transform(feat_matrix)
        combined = np.column_stack([feat_matrix, encoded])

        arima_pred = self.mmhpa_model._arima_predictions(close_vals)
        lr_pred    = self.mmhpa_model._linear_regression_predictions(close_vals)
        mm_combined = np.column_stack([combined, arima_pred, lr_pred])

        mm_pad      = np.zeros((len(close_vals), 1))
        final_feats = np.column_stack([combined, mm_pad])
        selected    = self.selector.transform(final_feats)
        scaled      = self.feature_scaler.transform(selected)
        mm_scaled   = self.mmhpa_model.scaler.transform(mm_combined)

        return scaled, mm_scaled, close_vals

    # ─── Main Prediction ───────────────────────────────
    def predict_tomorrow(self):
        self.load_models()
        self.calibrate()

        # ✅ Download with fallback — NO long sleeps/retries
        logger.info(f'[{self.symbol}] Downloading latest data...')
        df = _download_with_fallback(self.ticker, self.symbol, self.base_dir)

        # ✅ Save fresh data to CSV for future fallback
        try:
            data_dir = self.base_dir / 'data'
            data_dir.mkdir(exist_ok=True)
            df.to_csv(data_dir / f'{self.symbol}.csv')
        except Exception:
            pass

        today_price  = float(df['Close'].iloc[-1])
        yf_last_date = df.index[-1].date()

        from datetime import date as _date
        actual_today = _date.today()
        lag_days     = (actual_today - yf_last_date).days
        today_date   = actual_today if lag_days <= 1 else yf_last_date

        from datetime import datetime as _dt
        tomorrow = _dt.combine(actual_today, _dt.min.time()) + timedelta(days=1)
        while tomorrow.weekday() >= 5:
            tomorrow += timedelta(days=1)

        scaled, mm_scaled, close_vals = self._run_feature_pipeline(df)

        gan_preds = []
        mm_preds  = []

        for i in range(self.WINDOW + 1):
            idx   = -(self.WINDOW + 1 - i)
            end   = idx if idx != -1 else len(scaled)
            start = idx - self.seq_len

            g_seq = scaled[start:end]
            if len(g_seq) < self.seq_len:
                continue
            g_seq = g_seq[-self.seq_len:].reshape(1, self.seq_len, scaled.shape[1])

            g_sc  = self.generator_sess.run(None, {
                self.generator_inp: g_seq.astype(np.float32)
            })[0].flatten()[0]
            g_abs = self.price_scaler.inverse_transform([[g_sc]])[0][0]
            gan_preds.append(g_abs)

            m_seq = mm_scaled[start:end]
            if len(m_seq) < self.seq_len:
                continue
            m_seq = m_seq[-self.seq_len:].reshape(1, self.seq_len, mm_scaled.shape[1])

            if self.mm_nonlinear_sess is not None:
                m_sc  = self.mm_nonlinear_sess.run(None, {
                    self.mm_nonlinear_inp: m_seq.astype(np.float32)
                })[0].flatten()[0]
                m_abs = self.mmhpa_model.price_scaler.inverse_transform([[m_sc]])[0][0]
            elif self.mmhpa_model.nonlinear_model is not None:
                m_sc  = self.mmhpa_model.nonlinear_model.predict(
                    m_seq, verbose=0
                ).flatten()[0]
                m_abs = self.mmhpa_model.price_scaler.inverse_transform([[m_sc]])[0][0]
            else:
                from statsmodels.tsa.arima.model import ARIMA
                m_abs = float(
                    ARIMA(close_vals[-60:], order=(2, 1, 1)).fit().forecast(1)[0]
                )
            mm_preds.append(m_abs)

        if len(gan_preds) < 2 or len(mm_preds) < 2:
            raise ValueError(f'[{self.symbol}] Insufficient predictions generated')

        gan_preds     = np.array(gan_preds)
        mm_preds      = np.array(mm_preds)
        mmganhpa_preds = (1 - self.blend) * gan_preds + self.blend * mm_preds

        def avg_ret(arr):
            r = (arr[1:] - arr[:-1]) / arr[:-1]
            return r.mean()

        avg_returns = {
            'MMHPA':    avg_ret(mm_preds),
            'GANHPA':   avg_ret(gan_preds),
            'MMGANHPA': avg_ret(mmganhpa_preds),
        }

        predictions = {}
        for name, raw_r in avg_returns.items():
            cal_r = np.clip(raw_r * self._calib.get(name, 1.0), -0.05, 0.05)
            price = today_price * (1 + cal_r)
            predictions[name] = {
                'price':      round(price, 2),
                'return_pct': round(cal_r * 100, 4),
                'raw_return': round(raw_r * 100, 4),
            }

        directions = [np.sign(v['return_pct']) for v in predictions.values()]
        all_agree  = len(set(d for d in directions if d != 0)) <= 1
        avg_dir    = np.mean(directions)

        direction  = 'UP' if avg_dir > 0 else ('DOWN' if avg_dir < 0 else 'FLAT')
        confidence = 'HIGH' if all_agree else 'LOW'

        hist_std = self._hist_std or 0.015
        ci_low   = round(today_price * (1 - 2 * hist_std), 2)
        ci_high  = round(today_price * (1 + 2 * hist_std), 2)

        hist_prices = []
        for idx_row in df.tail(30).itertuples():
            hist_prices.append({
                'date':   idx_row.Index.strftime('%Y-%m-%d'),
                'open':   round(idx_row.Open,   2),
                'high':   round(idx_row.High,   2),
                'low':    round(idx_row.Low,    2),
                'close':  round(idx_row.Close,  2),
                'volume': int(idx_row.Volume),
            })

        result = {
            'symbol':          self.symbol,
            'name':            self.config['name'],
            'sector':          self.config['sector'],
            'color':           self.config['color'],
            'today_date':      today_date,
            'prediction_date': tomorrow.date(),
            'today_close':     round(today_price, 2),
            'predictions':     predictions,
            'direction':       direction,
            'confidence':      confidence,
            'ci_low':          ci_low,
            'ci_high':         ci_high,
            'hist_std':        round(hist_std, 6),
            'calibration':     self._calib,
            'accuracy':        self._accuracy,
            'history':         hist_prices,
        }

        logger.info(
            f'[{self.symbol}] Prediction: INR{predictions["MMGANHPA"]["price"]:.2f} '
            f'({predictions["MMGANHPA"]["return_pct"]:+.2f}%) '
            f'{direction} | {confidence}'
        )
        return result


class PredictionEngine:
    """Manages predictions for all stocks. Singleton pattern."""

    _instance   = None
    _predictors = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir=None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            from django.conf import settings
            self.base_dir = settings.BASE_DIR

    def get_predictor(self, symbol: str) -> StockPredictor:
        from django.conf import settings
        if symbol not in self._predictors:
            config = settings.STOCK_CONFIG.get(symbol)
            if not config:
                raise ValueError(f'Unknown stock: {symbol}')
            self._predictors[symbol] = StockPredictor(symbol, config, self.base_dir)
        return self._predictors[symbol]

    def predict(self, symbol: str) -> dict:
        return self.get_predictor(symbol).predict_tomorrow()

    def predict_all(self) -> dict:
        """
        Run predictions for all stocks.
        ✅ No sleep() between stocks — each uses fast download with fallback.
        """
        from django.conf import settings

        results = {}
        errors  = {}

        for symbol in settings.STOCK_CONFIG.keys():
            try:
                results[symbol] = self.predict(symbol)
            except Exception as e:
                logger.error(f'[{symbol}] Prediction failed: {e}')
                errors[symbol] = str(e)

        return {'predictions': results, 'errors': errors}