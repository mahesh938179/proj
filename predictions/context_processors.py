from django.conf import settings


def stock_list_context(request):
    """Make stock list available in all templates"""
    return {
        'all_stocks': [
            {
                'symbol': sym,
                'name': cfg['name'],
                'sector': cfg['sector'],
                'color': cfg['color'],
            }
            for sym, cfg in settings.STOCK_CONFIG.items()
        ]
    }
