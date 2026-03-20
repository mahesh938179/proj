from django.contrib import admin
from .models import Stock, StockPrediction, PriceHistory, PredictionAccuracy


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'name', 'sector', 'is_active', 'created_at']
    list_filter = ['sector', 'is_active']
    search_fields = ['symbol', 'name']


@admin.register(StockPrediction)
class StockPredictionAdmin(admin.ModelAdmin):
    list_display = [
        'stock', 'prediction_date', 'today_close',
        'mmganhpa_price', 'mmganhpa_return_pct',
        'direction', 'confidence', 'created_at',
    ]
    list_filter = ['direction', 'confidence', 'stock', 'prediction_date']
    date_hierarchy = 'prediction_date'
    readonly_fields = ['created_at', 'updated_at']


@admin.register(PriceHistory)
class PriceHistoryAdmin(admin.ModelAdmin):
    list_display = ['stock', 'date', 'open', 'high', 'low', 'close', 'volume']
    list_filter = ['stock']
    date_hierarchy = 'date'


@admin.register(PredictionAccuracy)
class PredictionAccuracyAdmin(admin.ModelAdmin):
    list_display = [
        'stock', 'date', 'predicted_direction',
        'actual_direction', 'is_correct', 'error_pct',
    ]
    list_filter = ['is_correct', 'stock']
    date_hierarchy = 'date'
