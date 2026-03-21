from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    # Pages
    path('', views.dashboard, name='dashboard'),
    path(
        'stock/<str:symbol>/',
        views.stock_detail,
        name='stock_detail',
    ),
    path('compare/', views.compare, name='compare'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),

    # API
    path(
        'api/predict/<str:symbol>/',
        views.api_predict,
        name='api_predict',
    ),
    path(
        'api/predict-all/',
        views.api_predict_all,
        name='api_predict_all',
    ),
    path(
        'api/stock/<str:symbol>/',
        views.api_stock_data,
        name='api_stock_data',
    ),
    path(
        'api/refresh/<str:symbol>/',
        views.api_refresh,
        name='api_refresh',
    ),
    path(
        'api/live-prices/',
        views.get_live_prices,
        name='api_live_prices',
    ),
    path('api/predict-status/', views.api_predict_status, name='api_predict_status'),
    # predictions/urls.py లో ఈ lines add చేయండి:

path('api/flutter/login/', views.api_flutter_login, name='api_flutter_login'),
path('api/flutter/dashboard/', views.api_flutter_dashboard, name='api_flutter_dashboard'),
path('api/flutter/stock/<str:symbol>/', views.api_flutter_stock_detail, name='api_flutter_stock_detail'),
path('api/flutter/live-prices/', views.get_live_prices, name='api_flutter_live'),
path('api/flutter/refresh/<str:symbol>/', views.api_refresh, name='api_flutter_refresh'),
]
