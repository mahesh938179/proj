from django.db import models
from django.utils import timezone


class Stock(models.Model):
    """Stock master data"""
    symbol = models.CharField(max_length=20, unique=True, db_index=True)
    ticker = models.CharField(max_length=30)
    name = models.CharField(max_length=200)
    sector = models.CharField(max_length=100)
    color = models.CharField(max_length=10, default='#1565C0')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['symbol']

    def __str__(self):
        return f"{self.symbol} — {self.name}"

    @property
    def latest_prediction(self):
        return self.predictions.order_by('-created_at').first()

    @property
    def latest_price(self):
        pred = self.latest_prediction
        return pred.today_close if pred else None


class StockPrediction(models.Model):
    """One prediction record per stock per day"""

    class Direction(models.TextChoices):
        UP = 'UP', '▲ UP'
        DOWN = 'DOWN', '▼ DOWN'
        FLAT = 'FLAT', '▬ FLAT'

    class Confidence(models.TextChoices):
        HIGH = 'HIGH', 'HIGH ✅'
        LOW = 'LOW', 'LOW ⚠️'

    stock = models.ForeignKey(
        Stock, on_delete=models.CASCADE, related_name='predictions'
    )
    prediction_date = models.DateField(
        help_text='The date being predicted (tomorrow)'
    )
    data_date = models.DateField(
        help_text='Latest data date used'
    )
    today_close = models.FloatField()

    # Model predictions
    mmhpa_price = models.FloatField()
    mmhpa_return_pct = models.FloatField()
    ganhpa_price = models.FloatField()
    ganhpa_return_pct = models.FloatField()
    mmganhpa_price = models.FloatField()
    mmganhpa_return_pct = models.FloatField()

    # Consensus
    direction = models.CharField(
        max_length=5, choices=Direction.choices
    )
    confidence = models.CharField(
        max_length=5, choices=Confidence.choices
    )

    # Confidence interval
    ci_low = models.FloatField()
    ci_high = models.FloatField()
    hist_std = models.FloatField()

    # Calibration info
    mmhpa_calib_scale = models.FloatField(default=1.0)
    ganhpa_calib_scale = models.FloatField(default=1.0)
    mmganhpa_calib_scale = models.FloatField(default=1.0)
    mmhpa_5d_accuracy = models.FloatField(default=0.0)
    ganhpa_5d_accuracy = models.FloatField(default=0.0)
    mmganhpa_5d_accuracy = models.FloatField(default=0.0)

    # Raw returns (before calibration)
    mmhpa_raw_return = models.FloatField(default=0.0)
    ganhpa_raw_return = models.FloatField(default=0.0)
    mmganhpa_raw_return = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-prediction_date', 'stock__symbol']
        unique_together = ['stock', 'prediction_date']
        indexes = [
            models.Index(fields=['prediction_date']),
            models.Index(fields=['stock', '-prediction_date']),
        ]

    def __str__(self):
        return (
            f"{self.stock.symbol} | {self.prediction_date} | "
            f"₹{self.mmganhpa_price:.2f} ({self.direction})"
        )

    @property
    def best_price(self):
        return self.mmganhpa_price

    @property
    def best_return(self):
        return self.mmganhpa_return_pct

    @property
    def direction_icon(self):
        icons = {'UP': '▲', 'DOWN': '▼', 'FLAT': '▬'}
        return icons.get(self.direction, '?')

    @property
    def direction_color(self):
        colors = {'UP': '#00C853', 'DOWN': '#FF1744', 'FLAT': '#FFC107'}
        return colors.get(self.direction, '#9E9E9E')


class PriceHistory(models.Model):
    """Historical OHLCV data (cached)"""
    stock = models.ForeignKey(
        Stock, on_delete=models.CASCADE, related_name='price_history'
    )
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        ordering = ['-date']
        unique_together = ['stock', 'date']
        indexes = [
            models.Index(fields=['stock', '-date']),
        ]

    def __str__(self):
        return f"{self.stock.symbol} | {self.date} | ₹{self.close:.2f}"


class PredictionAccuracy(models.Model):
    """Track prediction accuracy over time"""
    stock = models.ForeignKey(
        Stock, on_delete=models.CASCADE, related_name='accuracy_records'
    )
    date = models.DateField()
    predicted_direction = models.CharField(max_length=5)
    actual_direction = models.CharField(max_length=5)
    predicted_price = models.FloatField()
    actual_price = models.FloatField()
    is_correct = models.BooleanField()
    error_pct = models.FloatField()

    class Meta:
        ordering = ['-date']
        unique_together = ['stock', 'date']

    def __str__(self):
        status = '✅' if self.is_correct else '❌'
        return f"{status} {self.stock.symbol} | {self.date}"
