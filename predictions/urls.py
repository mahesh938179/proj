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
]
