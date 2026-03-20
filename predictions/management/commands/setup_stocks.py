"""
Initialize stock records in DB from settings.
Usage: python manage.py setup_stocks
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from predictions.models import Stock


class Command(BaseCommand):
    help = 'Initialize stock records from STOCK_CONFIG settings'

    def handle(self, *args, **options):
        for symbol, config in settings.STOCK_CONFIG.items():
            stock, created = Stock.objects.update_or_create(
                symbol=symbol,
                defaults={
                    'ticker': config['ticker'],
                    'name': config['name'],
                    'sector': config['sector'],
                    'color': config['color'],
                    'is_active': True,
                }
            )
            status = 'Created' if created else 'Updated'
            self.stdout.write(
                self.style.SUCCESS(f'{status}: {symbol} — {config["name"]}')
            )

        self.stdout.write(
            self.style.SUCCESS(
                f'\nDone! {len(settings.STOCK_CONFIG)} stocks configured.'
            )
        )
