"""
Generate a secure API token for production use.
Usage: python manage.py generate_token
"""
import secrets
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Generate a secure API token for production'

    def handle(self, *args, **options):
        token = secrets.token_hex(32)
        self.stdout.write(self.style.SUCCESS(
            f'\nGenerated API Token:\n  {token}\n\n'
            f'Add to your environment:\n'
            f'  export API_TOKEN="{token}"\n\n'
            f'Or add to .env file:\n'
            f'  API_TOKEN={token}\n\n'
            f'Use in API requests:\n'
            f'  curl -H "X-Api-Token: {token}" http://localhost:8000/api/predict/TCS/\n'
        ))
