import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'django-insecure-change-me-in-production-x7k!q2m'
)
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    # Our app
    'predictions.apps.PredictionsConfig',
    'accounts.apps.AccountsConfig',
    # CORS
    'corsheaders',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',  # ✅ CORS — must be before CommonMiddleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'stockpredictor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'predictions.context_processors.stock_list_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'stockpredictor.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ─── ML Config ──────────────────────────────────────────
ML_MODELS_DIR = BASE_DIR / 'outputs' / 'models'
ML_RESULTS_DIR = BASE_DIR / 'outputs' / 'results'
ML_GAN_DIR = BASE_DIR / 'outputs' / 'gan'
ML_MMHPA_DIR = BASE_DIR / 'outputs' / 'mmhpa'
ML_MODULES_DIR = BASE_DIR / 'ml_modules'
ML_DATA_DIR = BASE_DIR / 'data'

# API Security — set a strong random token in production
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
API_TOKEN = os.environ.get('API_TOKEN', '')   # empty = no auth in dev

# ─── Security Headers ────────────────────────────────────
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER   = True
X_FRAME_OPTIONS              = 'DENY'
REFERRER_POLICY              = 'strict-origin-when-cross-origin'

# In production (DEBUG=False), also set:
# SECURE_SSL_REDIRECT = True
# SESSION_COOKIE_SECURE = True
# CSRF_COOKIE_SECURE = True

# ─── Cache (In-memory for dev, Redis for prod) ───────────
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'stockai-cache',
    }
}

# Ensure required output directories exist on startup
for _d in [
    ML_MODELS_DIR / 'gan',
    ML_MODELS_DIR / 'mmhpa',
    ML_MODELS_DIR / 'autoencoder',
    ML_MODELS_DIR / 'selection',
    ML_MODELS_DIR / 'scalers',
    ML_RESULTS_DIR / 'predictions',
    ML_GAN_DIR,
    ML_MMHPA_DIR,
    ML_DATA_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

STOCK_CONFIG = {
    'TCS': {
        'ticker': 'TCS.NS',
        'name': 'Tata Consultancy Services',
        'sector': 'IT',
        'color': '#1565C0',
        'blend': 0.08,
    },
    'WIPRO': {
        'ticker': 'WIPRO.NS',
        'name': 'Wipro Limited',
        'sector': 'IT',
        'color': '#6A1B9A',
        'blend': 0.08,
    },
    'TATASTEEL': {
        'ticker': 'TATASTEEL.NS',
        'name': 'Tata Steel Limited',
        'sector': 'Metals',
        'color': '#E65100',
        'blend': 0.08,
    },
    'MARUTI': {
        'ticker': 'MARUTI.NS',
        'name': 'Maruti Suzuki India',
        'sector': 'Auto',
        'color': '#2E7D32',
        'blend': 0.08,
    },
    'AXISBANK': {
        'ticker': 'AXISBANK.NS',
        'name': 'Axis Bank Limited',
        'sector': 'Banking',
        'color': '#AD1457',
        'blend': 0.08,
    },
    'BHEL': {
        'ticker': 'BHEL.NS',
        'name': 'Bharat Heavy Electricals',
        'sector': 'Capital Goods',
        'color': '#00838F',
        'blend': 0.08,
    },
}

# Cache predictions for N minutes
PREDICTION_CACHE_MINUTES = 60  # 1 hour cache

# Gunicorn timeout (seconds) — set in start command too
GUNICORN_TIMEOUT = 120

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'predictions.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'predictions': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    },
}

# Create logs directory
(BASE_DIR / 'logs').mkdir(exist_ok=True)
(BASE_DIR / 'media' / 'charts').mkdir(parents=True, exist_ok=True)
# ─── Auth Config ─────────────────────────────────────────
LOGIN_REDIRECT_URL = 'predictions:dashboard'
LOGOUT_REDIRECT_URL = 'accounts:login'
LOGIN_URL = 'accounts:login'

# ─── CORS — Flutter app access ───────────────────────────
CORS_ALLOWED_ORIGINS = [
    'http://localhost:63676',
    'http://localhost:8080',
    'http://127.0.0.1:63676',
    'http://127.0.0.1:8080',
]
CORS_ALLOW_ALL_ORIGINS = True  # Flutter web dev లో అన్నీ allow
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ['GET', 'POST', 'OPTIONS']
CORS_ALLOW_HEADERS = ['content-type', 'x-api-token', 'authorization']