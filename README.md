# 📈 StockAI Predictor

A production-ready **Django web application** for real-time Indian stock market prediction powered by a custom hybrid deep learning architecture — **MMGAN-HPA** (Multi-Model GAN Hybrid Prediction Algorithm). It predicts next-day closing prices for 6 major NSE stocks using ensemble ML models, all served via ONNX Runtime — no TensorFlow/Keras required at runtime.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Supported Stocks](#supported-stocks)
4. [Project Structure](#project-structure)
5. [ML Architecture](#ml-architecture)
6. [Setup & Installation](#setup--installation)
7. [Environment Variables](#environment-variables)
8. [Running the App](#running-the-app)
9. [Management Commands](#management-commands)
10. [ONNX Conversion Pipeline](#onnx-conversion-pipeline)
11. [API Endpoints](#api-endpoints)
12. [Database Models](#database-models)
13. [Deployment (Render)](#deployment-render)
14. [Troubleshooting](#troubleshooting)

---

## Features

- 🤖 **Hybrid ML Predictions** — Three models (MM-HPA, GAN-HPA, MMGAN-HPA) ensemble for directional accuracy ~79.6%
- 📊 **Live Price Updates** — Polls NSE prices every 30–60 seconds via yfinance; updates ROI calculations live
- 📅 **Daily Predictions** — Next trading day price + 95% confidence interval
- 📈 **Interactive Charts** — Price history, prediction overlays, candlestick charts
- 🔍 **Stock Comparison** — Side-by-side predictions sorted by expected return
- 📋 **Accuracy Tracker** — Historical accuracy logs stored per stock
- 🏪 **PWA Ready** — Installable as a Progressive Web App; Android APK via PWABuilder/TWA
- 🔐 **Auth System** — Login-required dashboard, role-based access
- ⚡ **API Rate Limiting** — 10 requests/min per IP; optional API token authentication
- 📡 **Health Check Endpoint** — `/health/` for uptime monitoring

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Django 5.1+ |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **ML Inference** | ONNX Runtime 1.17+ |
| **Static Files** | WhiteNoise |
| **Data Source** | yfinance (NSE via Yahoo Finance) |
| **Feature Engineering** | scikit-learn, statsmodels, minisom, scipy |
| **Boosting** | XGBoost (feature selection) |
| **Serving** | Gunicorn (production) |
| **Deployment** | Render.com |

---

## Supported Stocks

| Symbol | Company | Sector | NSE Ticker |
|---|---|---|---|
| `TCS` | Tata Consultancy Services | IT | `TCS.NS` |
| `WIPRO` | Wipro Limited | IT | `WIPRO.NS` |
| `TATASTEEL` | Tata Steel Limited | Metals | `TATASTEEL.NS` |
| `MARUTI` | Maruti Suzuki India | Auto | `MARUTI.NS` |
| `AXISBANK` | Axis Bank Limited | Banking | `AXISBANK.NS` |
| `BHEL` | Bharat Heavy Electricals | Capital Goods | `BHEL.NS` |

---

## Project Structure

```
proj/
├── accounts/                   # User authentication app
│   ├── models.py               # User, DeletionRequest models
│   ├── views.py                # Login, register, profile views
│   ├── forms.py                # Auth forms
│   ├── decorators.py           # Role-based access decorators
│   └── management/commands/    # create_admin, set_role, generate_token
│
├── predictions/                # Core prediction app
│   ├── models.py               # Stock, StockPrediction, PriceHistory, PredictionAccuracy
│   ├── views.py                # Dashboard, stock detail, compare, history, API endpoints
│   ├── engine.py               # Core prediction engine (ONNX-based, no Keras at runtime)
│   ├── context_processors.py   # Injects stock list into all templates
│   ├── urls.py                 # URL routes
│   └── management/commands/    # setup_stocks, run_predictions, regenerate_calibration
│
├── ml_modules/                 # ML pipeline modules
│   ├── autoencoders.py         # Stacked autoencoder (ONNX inference mode)
│   ├── feature_extraction.py   # Technical indicators, Fourier, ARIMA, SOM features
│   ├── feature_selection.py    # XGBoost importance + PCA selector
│   ├── mm_hpa.py               # MM-HPA: ARIMA + Linear Regression + LSTM
│   ├── mmgan_hpa.py            # Full MMGAN-HPA ensemble
│   ├── stock_gan.py            # Stock GAN (LSTM generator + CNN discriminator)
│   └── prediction_service.py   # Full training pipeline service
│
├── outputs/                    # Generated model artifacts (git-ignored)
│   └── models/
│       ├── gan/                # *_generator.onnx, *_discriminator.keras
│       ├── mmhpa/              # *_mmhpa.pkl (Keras stripped), *_mmhpa_nonlinear.onnx
│       ├── autoencoder/        # *_autoencoder.pkl (Keras stripped), *_encoder.onnx
│       ├── selection/          # *_selector.pkl
│       └── scalers/            # *_feature_scaler.pkl, *_price_scaler.pkl
│
├── templates/                  # Django HTML templates
│   ├── dashboard.html          # Main dashboard
│   ├── stock_detail.html       # Per-stock detail + charts
│   ├── compare.html            # Stock comparison view
│   ├── history.html            # Prediction history + accuracy
│   ├── about.html              # About page
│   ├── partials/               # Reusable sidebar, navbar, etc.
│   └── pwa/                    # manifest.json, service-worker.js, assetlinks.json
│
├── static/                     # CSS, JS, images
│   └── css/style.css
│
├── stockpredictor/             # Django project config
│   ├── settings.py             # All settings + STOCK_CONFIG
│   ├── urls.py                 # Root URL conf + health check
│   ├── wsgi.py
│   └── asgi.py
│
├── data/                       # Cached CSV data per stock
├── logs/                       # predictions.log
│
├── convert_to_onnx.py          # Step 1: Convert GAN .keras → .onnx
├── strip_keras_from_pkl.py     # Step 2: Strip Keras LSTM from mmhpa.pkl
├── convert_autoencoders_to_onnx.py  # Step 3: Convert encoder → .onnx, strip pkl
│
├── requirements.txt
├── build.sh                    # Render deployment build script
├── manage.py
└── .env.example
```

---

## ML Architecture

The prediction system uses a 3-stage ensemble pipeline called **MMGAN-HPA**:

```
Raw Stock Data (OHLCV, 90 days)
        │
        ▼
┌─────────────────────────────────────────┐
│  PHASE 1: Feature Extraction            │
│  • 53 features: Technical Indicators,  │
│    Fourier Transform, ARIMA residuals, │
│    SOM anomaly scores                  │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  PHASE 2: Stacked Autoencoder (ONNX)    │
│  • Compresses 53 → 32 encoded features │
│  • Trained with [128, 64, 32] dims     │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐   ┌─────────────────────────┐
│  MM-HPA Path  │   │  GAN-HPA Path           │
│  • ARIMA      │   │  • LSTM Generator       │
│  • Linear Reg │   │    (3 layers, ONNX)     │
│  • LSTM ONNX  │   │  • CNN Discriminator    │
│    (optional) │   │    (training only)      │
└───────┬───────┘   └──────────┬──────────────┘
        │                      │
        └──────────┬───────────┘
                   │  blend = 0.08
                   ▼
        ┌─────────────────────┐
        │  MMGAN-HPA Ensemble │
        │  0.92×GAN + 0.08×MM │
        └──────────┬──────────┘
                   │
                   ▼
        5-Day Average Return Signal
        → Calibrated Price Prediction
        → Direction (UP/DOWN/FLAT)
        → 95% Confidence Interval
```

### Key Performance Metrics (from training)
| Model | 5-Day Direction Accuracy |
|---|---|
| MM-HPA | ~75.1% |
| GAN-HPA | ~79.1% |
| **MMGAN-HPA** | **~79.6%** |

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd proj
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
copy .env.example .env
# Edit .env with your values
```

### 5. Run database migrations
```bash
python manage.py migrate
```

### 6. Initialize stock data and admin
```bash
python manage.py setup_stocks        # Creates Stock records in DB
python manage.py create_admin        # Creates default admin user
python manage.py set_role MAHESH superadmin   # Promote user to superadmin
python manage.py generate_token      # Generates API token
```

### 7. Run first predictions
```bash
python manage.py run_predictions
```

---

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | ✅ | Django secret key (generate a strong random one for production) |
| `DEBUG` | ✅ | `True` for development, `False` for production |
| `ALLOWED_HOSTS` | ✅ | Comma-separated list of allowed hostnames |
| `API_TOKEN` | ⚠️ Optional | Token for API authentication. Empty = no auth (dev mode) |

**Generate a secret key:**
```bash
python -c "import secrets; print(secrets.token_hex(50))"
```

**Generate an API token:**
```bash
python manage.py generate_token
```

---

## Running the App

### Development
```bash
python manage.py runserver
```
Visit: `http://127.0.0.1:8000`

### Production (Gunicorn)
```bash
gunicorn stockpredictor.wsgi:application --bind 0.0.0.0:8000
```

---

## Management Commands

| Command | Description |
|---|---|
| `python manage.py setup_stocks` | Creates/updates all 6 Stock records in the database |
| `python manage.py run_predictions` | Runs full ML pipeline for all stocks, saves results to DB |
| `python manage.py create_admin` | Creates a default admin superuser |
| `python manage.py set_role <username> <role>` | Set a user's role (e.g. `superadmin`) |
| `python manage.py generate_token` | Generates and prints a secure API token |
| `python manage.py regenerate_calibration` | Re-runs calibration for all stocks from saved test data |
| `python manage.py migrate` | Apply database migrations |
| `python manage.py collectstatic` | Collect static files for production |

---

## ONNX Conversion Pipeline

The models were trained with TensorFlow/Keras. For deployment (where `keras` may not be available), all Keras models are converted to ONNX and stripped from pickle files.

> **This only needs to be done ONCE, or after retraining.**

### Run in this exact order:

```bash
# Step 1 — Convert GAN generator .keras files → .onnx
python convert_to_onnx.py

# Step 2 — Strip embedded Keras LSTM from mmhpa.pkl files
python strip_keras_from_pkl.py

# Step 3 — Convert autoencoder encoders → .onnx, strip from pkl
python convert_autoencoders_to_onnx.py
```

### What each script does:

| Script | Input | Output |
|---|---|---|
| `convert_to_onnx.py` | `outputs/models/gan/*_generator.keras` | `outputs/models/gan/*_generator.onnx` |
| `strip_keras_from_pkl.py` | `outputs/models/mmhpa/*_mmhpa.pkl` (with Keras LSTM inside) | Same pkls, `nonlinear_model = None` |
| `convert_autoencoders_to_onnx.py` | `outputs/models/autoencoder/*_autoencoder.pkl` (with Keras encoder) | `*_encoder.onnx` + pkls stripped |

### Why this matters

When models are trained, Keras model objects are embedded inside `.pkl` files via Python pickle. Unpickling these files on a system without standalone `keras` installed throws:
```
No module named 'keras'
```
The conversion pipeline removes all Keras objects from pkls and replaces them with ONNX sessions at load time.

### Runtime inference stack (after conversion)

```
engine.py loads:
  ├── *_generator.onnx          → GAN prediction (ONNX Runtime)
  ├── *_encoder.onnx            → Autoencoder encoding (ONNX Runtime)
  ├── *_mmhpa_nonlinear.onnx    → MMHPA LSTM (ONNX Runtime, if exists)
  ├── *_mmhpa.pkl               → ARIMA + Linear Regression (sklearn)
  ├── *_autoencoder.pkl         → ScikitLearn scaler only (Keras stripped)
  ├── *_selector.pkl            → XGBoost + PCA feature selector
  └── *_feature_scaler.pkl      → MinMaxScaler
      *_price_scaler.pkl        → MinMaxScaler
```

---

## API Endpoints

All API endpoints support rate limiting (10 req/min per IP). Token-protected endpoints require the `X-Api-Token` header.

| Method | URL | Auth | Description |
|---|---|---|---|
| `GET` | `/` | Login | Dashboard (all stocks) |
| `GET` | `/stock/<symbol>/` | Login | Stock detail view |
| `GET` | `/compare/` | Login | Side-by-side stock comparison |
| `GET` | `/history/` | Login | Prediction history + accuracy |
| `GET` | `/about/` | Public | About page |
| `GET` | `/health/` | Public | Health check (`{"status": "ok"}`) |
| `GET` | `/api/predict/<symbol>/` | Token | Run fresh prediction for one stock |
| `GET` | `/api/predict-all/` | Token | Run fresh predictions for all stocks |
| `GET` | `/api/live-prices/` | None | Get current NSE prices (used for live updates) |
| `GET` | `/api/stock/<symbol>/` | None | Latest DB data for a stock (AJAX) |
| `GET` | `/api/refresh/<symbol>/` | Token | Force refresh prediction + return JSON |
| `GET` | `/admin/` | Superuser | Django admin |

### API Token Usage
```bash
curl -H "X-Api-Token: your_token_here" \
     https://yourdomain.com/api/predict/TCS/
```

### Example API Response (`/api/predict/TCS/`)
```json
{
  "status": "success",
  "data": {
    "symbol": "TCS",
    "name": "Tata Consultancy Services",
    "today_close": 3850.25,
    "prediction_date": "2026-03-21",
    "predictions": {
      "MMHPA":    { "price": 3862.10, "return_pct": 0.3073 },
      "GANHPA":   { "price": 3871.45, "return_pct": 0.5505 },
      "MMGANHPA": { "price": 3870.70, "return_pct": 0.5329 }
    },
    "direction": "UP",
    "confidence": "HIGH",
    "ci_low": 3754.69,
    "ci_high": 3945.81
  }
}
```

---

## Database Models

### `Stock`
| Field | Type | Description |
|---|---|---|
| `symbol` | CharField | Unique stock symbol (e.g. `TCS`) |
| `ticker` | CharField | Yahoo Finance ticker (e.g. `TCS.NS`) |
| `name` | CharField | Full company name |
| `sector` | CharField | Business sector |
| `color` | CharField | Hex color for UI |
| `is_active` | BooleanField | Whether to include in predictions |

### `StockPrediction`
| Field | Type | Description |
|---|---|---|
| `stock` | FK → Stock | Which stock |
| `prediction_date` | DateField | Date being predicted |
| `today_close` | FloatField | Last known close price |
| `mmhpa_price` | FloatField | MM-HPA predicted price |
| `ganhpa_price` | FloatField | GAN-HPA predicted price |
| `mmganhpa_price` | FloatField | MMGAN-HPA predicted price (best) |
| `direction` | Choice | `UP` / `DOWN` / `FLAT` |
| `confidence` | Choice | `HIGH` / `LOW` |
| `ci_low`, `ci_high` | FloatField | 95% confidence interval |
| `mmganhpa_5d_accuracy` | FloatField | Model's 5-day direction accuracy |

### `PriceHistory`
OHLCV data cached from yfinance. Updated on each prediction run and via the live price API.

### `PredictionAccuracy`
Auto-populated when the actual close for a previously-predicted date becomes known. Tracks `is_correct` (direction match) and `error_pct`.

---

## Deployment (Render)

### Build Command
```bash
./build.sh
```

The `build.sh` script:
1. `pip install -r requirements.txt`
2. `python manage.py collectstatic --no-input`
3. `python manage.py migrate`
4. `python manage.py setup_stocks`
5. `python manage.py generate_token`
6. `python manage.py create_admin`
7. `python manage.py set_role MAHESH superadmin`

### Start Command
```bash
gunicorn stockpredictor.wsgi:application
```

### Required Environment Variables on Render
```
SECRET_KEY=<strong-random-key>
DEBUG=False
ALLOWED_HOSTS=yourapp.onrender.com
API_TOKEN=<generated-token>
```

### Production Security Checklist
- [ ] `DEBUG=False`
- [ ] Strong `SECRET_KEY` (50+ chars)
- [ ] `ALLOWED_HOSTS` set to your domain only
- [ ] `API_TOKEN` generated and stored securely
- [ ] WhiteNoise serving static files (already configured)
- [ ] `SECURE_SSL_REDIRECT=True` (uncomment in settings.py)
- [ ] `SESSION_COOKIE_SECURE=True`
- [ ] `CSRF_COOKIE_SECURE=True`

---

## Troubleshooting

### `No module named 'keras'` at startup
The `.pkl` model files still contain embedded Keras objects. Run the ONNX conversion pipeline in order:
```bash
python convert_to_onnx.py
python strip_keras_from_pkl.py
python convert_autoencoders_to_onnx.py
```

### `ONNX model not found: outputs/models/gan/TCS_generator.onnx`
The ONNX files are missing. Run `convert_to_onnx.py` first.

### `Encoder ONNX not found` warning in logs
Run `convert_autoencoders_to_onnx.py` to generate `*_encoder.onnx` files.

### Predictions show stale prices after browser refresh
The live price API (`/api/live-prices/`) updates the DB every 30–60 seconds. Ensure your frontend JavaScript is calling this endpoint to poll live prices. The `today_close` in the DB is updated on each live poll.

### `ALLOWED_HOSTS` error in production
Add your Render domain to `ALLOWED_HOSTS` in the environment variable:
```
ALLOWED_HOSTS=yourapp.onrender.com,localhost
```

### Static files not loading in production
Run `python manage.py collectstatic` and ensure `DEBUG=False`. WhiteNoise is configured to serve static files automatically.

---

## Logs

Application logs are written to `logs/predictions.log` and also to the console. Log level is `INFO` for the `predictions` logger.

```bash
# Tail logs in real-time
Get-Content logs\predictions.log -Wait -Tail 50   # Windows
tail -f logs/predictions.log                       # Linux/macOS
```

---

## License

This project is for educational and research purposes.
