import json
import logging
import yfinance as yf
from datetime import date, timedelta
from functools import wraps

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count, Q
from django.contrib.auth.decorators import login_required

from .models import Stock, StockPrediction, PriceHistory, PredictionAccuracy
from .engine import PredictionEngine

logger = logging.getLogger('predictions')

# ─── API Token Auth ──────────────────────────────────────
def require_api_token(view_func):
    """
    Simple token-based auth for API endpoints.
    Pass header:  X-API-Token: <value from settings.API_TOKEN>
    Or skip auth if API_TOKEN not set in settings (dev mode).
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        expected = getattr(settings, 'API_TOKEN', None)
        if expected:
            token = request.headers.get('X-Api-Token', '')
            if token != expected:
                return JsonResponse(
                    {'error': 'Unauthorized — invalid API token'},
                    status=401
                )
        return view_func(request, *args, **kwargs)
    return wrapper


# ─── Simple In-Memory Rate Limiter ──────────────────────
import time
from collections import defaultdict

_rate_store = defaultdict(list)
_RATE_LIMIT = 10        # requests
_RATE_WINDOW = 60       # seconds

def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < _RATE_WINDOW]
    if len(_rate_store[ip]) >= _RATE_LIMIT:
        return True
    _rate_store[ip].append(now)
    return False


def rate_limit_api(view_func):
    """Rate limit: max 10 API calls per IP per 60 seconds."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        ip = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', ''))
        ip = ip.split(',')[0].strip()
        if _is_rate_limited(ip):
            return JsonResponse(
                {'error': 'Rate limit exceeded — max 10 requests/minute'},
                status=429
            )
        return view_func(request, *args, **kwargs)
    return wrapper


# ─── Dashboard ──────────────────────────────────────────
@login_required
def dashboard(request):
    """Main dashboard — all stocks at a glance"""
    stocks = Stock.objects.filter(is_active=True).prefetch_related(
        'predictions'
    )

    stock_data = []
    for stock in stocks:
        # Always get the most recent prediction record (by created_at, not prediction_date)
        pred = stock.predictions.order_by('-created_at').first()
        history = stock.price_history.order_by('-date')[:30]

        # current_price: use pred.today_close (kept live by /api/live-prices/)
        # This is always the freshest price saved to DB
        current_price = pred.today_close if pred else None

        # Build price list oldest→newest for chart
        prices = list(history.values_list('close', flat=True))[::-1]

        if not current_price and prices:
            current_price = prices[-1]

        day_change = 0
        week_change = 0

        if current_price and len(prices) >= 2:
            prev_close = prices[-2]
            day_change = ((current_price - prev_close) / prev_close) * 100

        if current_price and len(prices) >= 5:
            five_days_ago = prices[-5]
            week_change = ((current_price - five_days_ago) / five_days_ago) * 100

        stock_data.append({
            'stock'       : stock,
            'prediction'  : pred,
            'current_price': current_price,
            'day_change'  : round(day_change, 2),
            'week_change' : round(week_change, 2),
            'prices': json.dumps(
                prices[-14:] if not current_price or (prices and abs(prices[-1] - current_price) < 0.01)
                else prices[-14:] + [current_price]
            ),
            'dates': json.dumps([
                h.date.strftime('%d %b')
                for h in sorted(list(history), key=lambda x: x.date)[:15]
            ]),
        })

    # Summary stats
    total_predictions = StockPrediction.objects.count()
    up_count = StockPrediction.objects.filter(
        prediction_date=date.today(),
        direction='UP'
    ).count()
    down_count = StockPrediction.objects.filter(
        prediction_date=date.today(),
        direction='DOWN'
    ).count()

    # Overall accuracy
    accuracy_records = PredictionAccuracy.objects.all()
    overall_accuracy = 0
    if accuracy_records.exists():
        overall_accuracy = round(
            accuracy_records.filter(is_correct=True).count()
            / accuracy_records.count() * 100, 1
        )
    else:
        # Fallback: Use average of model's reported 5-day accuracy if no records yet
        avg_acc = StockPrediction.objects.filter(
            prediction_date=date.today()
        ).aggregate(Avg('mmganhpa_5d_accuracy'))['mmganhpa_5d_accuracy__avg']
        
        if not avg_acc:
            # Try latest predictions overall
            avg_acc = StockPrediction.objects.all().order_by('-prediction_date')[:6] \
                .aggregate(Avg('mmganhpa_5d_accuracy'))['mmganhpa_5d_accuracy__avg']

        if avg_acc:
            overall_accuracy = round(avg_acc * 100, 1)

    context = {
        'stock_data': stock_data,
        'total_predictions': total_predictions,
        'up_count': up_count,
        'down_count': down_count,
        'overall_accuracy': overall_accuracy,
        'market_status': _get_market_status(),
    }
    return render(request, 'dashboard.html', context)


# ─── Stock Detail ───────────────────────────────────────
@login_required
def stock_detail(request, symbol):
    """Detailed view for a single stock"""
    symbol = symbol.upper()
    stock = get_object_or_404(Stock, symbol=symbol, is_active=True)

    # Latest prediction
    prediction = stock.predictions.order_by('-prediction_date').first()

    # Historical predictions (last 30)
    past_predictions = stock.predictions.order_by('-prediction_date')[:30]

    # Price history
    history = stock.price_history.order_by('date')[:60]
    history_data = {
        'dates': [h.date.strftime('%Y-%m-%d') for h in history],
        'closes': [round(h.close, 2) for h in history],
        'highs': [round(h.high, 2) for h in history],
        'lows': [round(h.low, 2) for h in history],
        'volumes': [h.volume for h in history],
    }

    # Prediction overlay
    pred_overlay = {
        'dates': [],
        'mmhpa': [],
        'ganhpa': [],
        'mmganhpa': [],
    }
    for p in sorted(list(past_predictions), key=lambda x: x.prediction_date):
        pred_overlay['dates'].append(
            p.prediction_date.strftime('%Y-%m-%d')
        )
        pred_overlay['mmhpa'].append(round(p.mmhpa_price, 2))
        pred_overlay['ganhpa'].append(round(p.ganhpa_price, 2))
        pred_overlay['mmganhpa'].append(round(p.mmganhpa_price, 2))

    # Accuracy stats
    accuracy_data = list(stock.accuracy_records.order_by('-date')[:30])
    total_count = len(accuracy_data)
    correct_count = len([r for r in accuracy_data if r.is_correct])
    accuracy_pct = (
        round(correct_count / total_count * 100, 1)
        if total_count > 0 else 0
    )

    # Model-wise confidence
    model_stats = {
        'MMHPA': {
            'accuracy': (
                round(prediction.mmhpa_5d_accuracy * 100, 1)
                if prediction else 0
            ),
            'calib': (
                round(prediction.mmhpa_calib_scale, 4)
                if prediction else 1.0
            ),
        },
        'GANHPA': {
            'accuracy': (
                round(prediction.ganhpa_5d_accuracy * 100, 1)
                if prediction else 0
            ),
            'calib': (
                round(prediction.ganhpa_calib_scale, 4)
                if prediction else 1.0
            ),
        },
        'MMGANHPA': {
            'accuracy': (
                round(prediction.mmganhpa_5d_accuracy * 100, 1)
                if prediction else 0
            ),
            'calib': (
                round(prediction.mmganhpa_calib_scale, 4)
                if prediction else 1.0
            ),
        },
    }

    config = settings.STOCK_CONFIG.get(symbol, {})

    context = {
        'stock': stock,
        'prediction': prediction,
        'past_predictions': past_predictions,
        'history_json': json.dumps(history_data),
        'pred_overlay_json': json.dumps(pred_overlay),
        'accuracy_pct': accuracy_pct,
        'correct_count': correct_count,
        'total_count': total_count,
        'accuracy_data': accuracy_data,
        'model_stats': model_stats,
        'config': config,
    }
    return render(request, 'stock_detail.html', context)


# ─── Compare Stocks ─────────────────────────────────────
@login_required
def compare(request):
    """Side-by-side comparison of all stocks"""
    stocks = Stock.objects.filter(is_active=True)
    comparison = []

    for stock in stocks:
        pred = stock.predictions.order_by('-prediction_date').first()
        if pred:
            comparison.append({
                'stock': stock,
                'prediction': pred,
                'best_return': pred.mmganhpa_return_pct,
                'direction': pred.direction,
                'confidence': pred.confidence,
            })

    # Sort by predicted return (best opportunity first)
    comparison.sort(key=lambda x: x['best_return'], reverse=True)

    # Chart data
    chart_data = {
        'labels': [c['stock'].symbol for c in comparison],
        'returns': [round(c['best_return'], 2) for c in comparison],
        'colors': [
            '#00C853' if c['best_return'] > 0 else '#FF1744'
            for c in comparison
        ],
        'prices': [
            round(c['prediction'].today_close, 2) for c in comparison
        ],
        'predicted': [
            round(c['prediction'].mmganhpa_price, 2)
            for c in comparison
        ],
    }

    context = {
        'comparison': comparison,
        'chart_data_json': json.dumps(chart_data),
    }
    return render(request, 'compare.html', context)


# ─── History ────────────────────────────────────────────
@login_required
def history(request):
    """Prediction history + accuracy tracker"""
    symbol = request.GET.get('stock', None)
    days = int(request.GET.get('days', 30))

    stocks = Stock.objects.filter(is_active=True)

    if symbol:
        predictions = StockPrediction.objects.filter(
            stock__symbol=symbol.upper()
        ).order_by('-prediction_date')[:days]
    else:
        predictions = StockPrediction.objects.all().order_by(
            '-prediction_date'
        )[:days * 6]

    # Accuracy summary per stock
    accuracy_summary = []
    for stock in stocks:
        records = stock.accuracy_records.all()
        total = records.count()
        correct = records.filter(is_correct=True).count()
        accuracy_summary.append({
            'stock': stock,
            'total': total,
            'correct': correct,
            'accuracy': round(correct / total * 100, 1) if total > 0 else 0,
        })

    context = {
        'predictions': predictions,
        'stocks': stocks,
        'selected_stock': symbol,
        'days': days,
        'accuracy_summary': accuracy_summary,
    }
    return render(request, 'history.html', context)


# ─── About ──────────────────────────────────────────────
def about(request):
    return render(request, 'about.html')


# ─── API Endpoints ──────────────────────────────────────
@require_GET
@rate_limit_api
@require_api_token
def api_predict(request, symbol):
    """Run fresh prediction for a stock — with cache"""
    symbol = symbol.upper()

    if symbol not in settings.STOCK_CONFIG:
        return JsonResponse(
            {'error': f'Unknown stock: {symbol}'}, status=404
        )

    from django.core.cache import cache
    cache_key = f'pred_result_{symbol}'
    cached = cache.get(cache_key)
    if cached:
        logger.info(f'[{symbol}] Returning cached prediction')
        return JsonResponse({'status': 'success', 'data': cached, 'cached': True})

    try:
        engine = PredictionEngine()
        result = engine.predict(symbol)
        _save_prediction(result)

        result_dict = {
            'symbol': result['symbol'],
            'name': result['name'],
            'today_close': result['today_close'],
            'prediction_date': str(result['prediction_date']),
            'predictions': result['predictions'],
            'direction': result['direction'],
            'confidence': result['confidence'],
            'ci_low': result['ci_low'],
            'ci_high': result['ci_high'],
        }
        timeout_mins = getattr(settings, 'PREDICTION_CACHE_MINUTES', 30)
        cache.set(cache_key, result_dict, timeout=timeout_mins * 60)

        return JsonResponse({'status': 'success', 'data': result_dict})
    except Exception as e:
        logger.error(f'API prediction failed for {symbol}: {e}')
        return JsonResponse(
            {'status': 'error', 'message': str(e)}, status=500
        )


@require_GET
@rate_limit_api
@require_api_token
def api_predict_all(request):
    """Run predictions for all stocks — background threading to avoid timeout"""
    import threading
    from django.core.cache import cache

    # Check if already running
    if cache.get('predict_all_running'):
        return JsonResponse({
            'status': 'running',
            'message': 'Predictions already in progress, please wait...'
        })

    def _run_all():
        cache.set('predict_all_running', True, timeout=300)
        try:
            engine = PredictionEngine()
            results = engine.predict_all()
            for symbol, result in results['predictions'].items():
                _save_prediction(result)
                # Cache each result
                timeout_mins = getattr(settings, 'PREDICTION_CACHE_MINUTES', 30)
                cache.set(f'pred_result_{symbol}', {
                    'symbol': result['symbol'],
                    'name': result['name'],
                    'today_close': result['today_close'],
                    'prediction_date': str(result['prediction_date']),
                    'predictions': result['predictions'],
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'ci_low': result['ci_low'],
                    'ci_high': result['ci_high'],
                }, timeout=timeout_mins * 60)
            cache.set('predict_all_done', True, timeout=300)
            logger.info('predict_all background task completed')
        except Exception as e:
            logger.error(f'predict_all background failed: {e}')
        finally:
            cache.delete('predict_all_running')

    thread = threading.Thread(target=_run_all, daemon=True)
    thread.start()

    return JsonResponse({
        'status': 'success',
        'message': 'Predictions started in background. Refresh page in ~60 seconds.',
    })


@require_GET
@rate_limit_api
def get_live_prices(request):
    """
    High-speed API to fetch ONLY current prices.
    Used for live dashboard updates every 30-60s.
    """
    results = {}
    for symbol, config in settings.STOCK_CONFIG.items():
        try:
            ticker = yf.Ticker(config['ticker'])
            fi    = ticker.fast_info
            price = fi.get('last_price') or fi.get('lastPrice') or fi.get('regularMarketPrice')
            if price is None:
                raise KeyError('last_price not found in fast_info')
            price = round(float(price), 2)
            results[symbol] = price

            # Update today_close + return_pct in the latest prediction
            pred = StockPrediction.objects.filter(
                stock__symbol=symbol
            ).order_by('-created_at').first()

            if pred and abs(pred.today_close - price) > 0.01:
                pred.today_close          = price
                pred.mmhpa_return_pct     = round((pred.mmhpa_price    - price) / price * 100, 4)
                pred.ganhpa_return_pct    = round((pred.ganhpa_price   - price) / price * 100, 4)
                pred.mmganhpa_return_pct  = round((pred.mmganhpa_price - price) / price * 100, 4)
                pred.direction = (
                    'UP'   if pred.mmganhpa_price > price else
                    'DOWN' if pred.mmganhpa_price < price else 'FLAT'
                )
                pred.save(update_fields=[
                    'today_close', 'mmhpa_return_pct',
                    'ganhpa_return_pct', 'mmganhpa_return_pct', 'direction',
                ])

                # Also update PriceHistory for today so reload shows fresh price
                from datetime import date as _date
                stock_obj = pred.stock
                PriceHistory.objects.update_or_create(
                    stock=stock_obj,
                    date=_date.today(),
                    defaults={
                        'open'  : fi.get('open', price),
                        'high'  : fi.get('day_high', price),
                        'low'   : fi.get('day_low',  price),
                        'close' : price,
                        'volume': int(fi.get('three_month_average_volume', 0) or 0),
                    }
                )

        except Exception as e:
            logger.warning(f"Live Price sync failed for {symbol}: {e}")
            continue
            
    return JsonResponse({
        'status': 'success',
        'prices': results,
        'timestamp': timezone.now().strftime('%H:%M:%S')
    })


@require_GET
def api_stock_data(request, symbol):
    """Get latest stock data for AJAX updates"""
    symbol = symbol.upper()
    stock = get_object_or_404(Stock, symbol=symbol)
    pred = stock.predictions.order_by('-prediction_date').first()

    if not pred:
        return JsonResponse({'error': 'No predictions yet'}, status=404)

    history = stock.price_history.order_by('date')[:30]

    return JsonResponse({
        'symbol': symbol,
        'name': stock.name,
        'current_price': pred.today_close,
        'predicted_price': pred.mmganhpa_price,
        'return_pct': pred.mmganhpa_return_pct,
        'direction': pred.direction,
        'confidence': pred.confidence,
        'prediction_date': str(pred.prediction_date),
        'ci_low': pred.ci_low,
        'ci_high': pred.ci_high,
        'history': {
            'dates': [
                h.date.strftime('%Y-%m-%d') for h in history
            ],
            'prices': [round(h.close, 2) for h in history],
        },
    })


@require_GET
@rate_limit_api
@require_api_token
def api_refresh(request, symbol):
    """Trigger fresh prediction in background — returns immediately"""
    import threading
    from django.core.cache import cache

    symbol = symbol.upper()
    if symbol not in settings.STOCK_CONFIG:
        return JsonResponse({'error': 'Unknown stock'}, status=404)

    # Check if already running for this symbol
    if cache.get(f'refresh_running_{symbol}'):
        return JsonResponse({
            'status': 'running',
            'message': f'{symbol} prediction already in progress...'
        })

    def _run(sym):
        cache.set(f'refresh_running_{sym}', True, timeout=180)
        try:
            engine = PredictionEngine()
            result = engine.predict(sym)
            _save_prediction(result)
            # Invalidate old cache
            cache.delete(f'pred_result_{sym}')
            logger.info(f'[{sym}] Background refresh complete')
        except Exception as e:
            logger.error(f'[{sym}] Background refresh failed: {e}')
        finally:
            cache.delete(f'refresh_running_{sym}')

    threading.Thread(target=_run, args=(symbol,), daemon=True).start()

    return JsonResponse({
        'status': 'success',
        'message': f'{symbol} prediction started. Refresh page in ~60 seconds.',
    })


# ─── Helpers ────────────────────────────────────────────
def _save_prediction(result):
    """Save prediction result to database"""
    stock, _ = Stock.objects.get_or_create(
        symbol=result['symbol'],
        defaults={
            'ticker': settings.STOCK_CONFIG[result['symbol']]['ticker'],
            'name': result['name'],
            'sector': result['sector'],
            'color': result['color'],
        }
    )

    # Update stock info
    stock.name = result['name']
    stock.sector = result['sector']
    stock.color = result['color']
    stock.save()

    # Save/update prediction
    preds = result['predictions']
    pred, created = StockPrediction.objects.update_or_create(
        stock=stock,
        prediction_date=result['prediction_date'],
        defaults={
            'data_date': result['today_date'],
            'today_close': result['today_close'],
            'mmhpa_price': preds['MMHPA']['price'],
            'mmhpa_return_pct': preds['MMHPA']['return_pct'],
            'ganhpa_price': preds['GANHPA']['price'],
            'ganhpa_return_pct': preds['GANHPA']['return_pct'],
            'mmganhpa_price': preds['MMGANHPA']['price'],
            'mmganhpa_return_pct': preds['MMGANHPA']['return_pct'],
            'direction': result['direction'],
            'confidence': result['confidence'],
            'ci_low': result['ci_low'],
            'ci_high': result['ci_high'],
            'hist_std': result['hist_std'],
            'mmhpa_calib_scale': result['calibration'].get('MMHPA', 1.0),
            'ganhpa_calib_scale': result['calibration'].get('GANHPA', 1.0),
            'mmganhpa_calib_scale': result['calibration'].get(
                'MMGANHPA', 1.0
            ),
            'mmhpa_5d_accuracy': result['accuracy'].get('MMHPA', 0.5),
            'ganhpa_5d_accuracy': result['accuracy'].get('GANHPA', 0.5),
            'mmganhpa_5d_accuracy': result['accuracy'].get(
                'MMGANHPA', 0.5
            ),
            'mmhpa_raw_return': preds['MMHPA']['raw_return'],
            'ganhpa_raw_return': preds['GANHPA']['raw_return'],
            'mmganhpa_raw_return': preds['MMGANHPA']['raw_return'],
        }
    )

    # ─── Live Accuracy Tracking ───
    # Look for a previous prediction that was targeting TODAY
    prev_pred = StockPrediction.objects.filter(
        stock=stock,
        prediction_date=result['today_date']
    ).first()

    if prev_pred:
        # We finally have actual data to evaluate this prediction!
        actual_close = result['today_close']
        actual_dir = 'UP' if actual_close > prev_pred.today_close else 'DOWN'
        if abs(actual_close - prev_pred.today_close) < 0.001:
            actual_dir = 'FLAT'
            
        PredictionAccuracy.objects.update_or_create(
            stock=stock,
            date=result['today_date'],
            defaults={
                'predicted_direction': prev_pred.direction,
                'actual_direction': actual_dir,
                'predicted_price': prev_pred.mmganhpa_price,
                'actual_price': actual_close,
                'is_correct': (prev_pred.direction == actual_dir),
                'error_pct': abs(actual_close - prev_pred.mmganhpa_price) / actual_close * 100
            }
        )

    # Save price history
    for hp in result.get('history', []):
        PriceHistory.objects.update_or_create(
            stock=stock,
            date=hp['date'],
            defaults={
                'open': hp['open'],
                'high': hp['high'],
                'low': hp['low'],
                'close': hp['close'],
                'volume': hp['volume'],
            }
        )

    return pred


def _prediction_to_dict(result):
    """Convert prediction result to JSON-serializable dict"""
    return {
        'symbol': result['symbol'],
        'name': result['name'],
        'today_close': result['today_close'],
        'prediction_date': str(result['prediction_date']),
        'predictions': result['predictions'],
        'direction': result['direction'],
        'confidence': result['confidence'],
        'ci_low': result['ci_low'],
        'ci_high': result['ci_high'],
    }


def _get_market_status():
    """Check if Indian market is open"""
    now = timezone.localtime()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute

    if weekday >= 5:
        return {'status': 'CLOSED', 'label': 'Weekend', 'color': '#FF9800'}

    market_open = (hour == 9 and minute >= 15) or (10 <= hour <= 14)
    market_close_time = hour == 15 and minute <= 30

    if market_open or market_close_time:
        return {'status': 'OPEN', 'label': 'Market Open', 'color': '#00C853'}
    elif hour < 9 or (hour == 9 and minute < 15):
        return {
            'status': 'PRE',
            'label': 'Pre-Market',
            'color': '#FFC107',
        }
    else:
        return {
            'status': 'CLOSED',
            'label': 'Market Closed',
            'color': '#FF5722',
        }


# ─── Error views ────────────────────────────────────────
def error_404(request, exception):
    return render(request, 'errors/404.html', status=404)