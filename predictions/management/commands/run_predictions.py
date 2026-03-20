"""
Management command to run predictions for all stocks.
Usage: python manage.py run_predictions
       python manage.py run_predictions --stock TCS
       python manage.py run_predictions --stock TCS WIPRO
"""

import time
import logging
from datetime import date

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from predictions.engine import PredictionEngine
from predictions.models import Stock, StockPrediction, PriceHistory

logger = logging.getLogger('predictions')


class Command(BaseCommand):
    help = 'Run ML predictions for configured stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stock',
            nargs='+',
            type=str,
            help='Specific stock symbols (e.g., TCS WIPRO)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-prediction even if today already predicted',
        )

    def handle(self, *args, **options):
        symbols = options.get('stock') or list(settings.STOCK_CONFIG.keys())
        symbols = [s.upper() for s in symbols]
        force = options.get('force', False)

        # Validate symbols
        for sym in symbols:
            if sym not in settings.STOCK_CONFIG:
                raise CommandError(
                    f'Unknown stock: {sym}. '
                    f'Available: {", ".join(settings.STOCK_CONFIG.keys())}'
                )

        self.stdout.write(
            self.style.HTTP_INFO(
                f'\n{"="*60}\n'
                f'  Stock Prediction Engine v2\n'
                f'  Stocks: {", ".join(symbols)}\n'
                f'  Date: {date.today()}\n'
                f'{"="*60}\n'
            )
        )

        engine = PredictionEngine()
        success = 0
        failed = 0
        skipped = 0

        for i, symbol in enumerate(symbols, 1):
            self.stdout.write(
                f'\n[{i}/{len(symbols)}] Processing {symbol}...'
            )

            # Check if already predicted today (unless --force)
            if not force:
                stock_obj = Stock.objects.filter(symbol=symbol).first()
                if stock_obj:
                    existing = StockPrediction.objects.filter(
                        stock=stock_obj,
                        data_date=date.today(),
                    ).exists()
                    if existing:
                        self.stdout.write(
                            self.style.WARNING(
                                f'  [SKIP] {symbol}: Already predicted today '
                                f'(use --force to re-run)'
                            )
                        )
                        skipped += 1
                        continue

            start = time.time()
            try:
                result = engine.predict(symbol)

                # Save to database (reusing view helper)
                from predictions.views import _save_prediction
                _save_prediction(result)

                elapsed = time.time() - start
                pred = result['predictions']['MMGANHPA']
                direction = result['direction']
                conf = result['confidence']

                self.stdout.write(
                    self.style.SUCCESS(
                        f'  [OK] {symbol}: INR{pred["price"]:.2f} '
                        f'({pred["return_pct"]:+.2f}%) '
                        f'{direction} | {conf} '
                        f'[{elapsed:.1f}s]'
                    )
                )
                success += 1

            except Exception as e:
                elapsed = time.time() - start
                self.stdout.write(
                    self.style.ERROR(
                        f'  [ERROR] {symbol}: {str(e)} [{elapsed:.1f}s]'
                    )
                )
                logger.error(f'Prediction failed for {symbol}: {e}')
                failed += 1

        # Summary
        self.stdout.write(
            self.style.HTTP_INFO(
                f'\n{"="*60}\n'
                f'  Summary: {success} success, '
                f'{failed} failed, {skipped} skipped\n'
                f'{"="*60}\n'
            )
        )

        if failed > 0:
            self.stdout.write(
                self.style.WARNING(
                    '  Check logs/predictions.log for details'
                )
            )
