from django import template

register = template.Library()


@register.filter
def abs_val(value):
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return value


@register.filter
def direction_icon(value):
    icons = {'UP': '▲', 'DOWN': '▼', 'FLAT': '▬'}
    return icons.get(value, '?')


@register.filter
def direction_class(value):
    classes = {
        'UP': 'text-success',
        'DOWN': 'text-danger',
        'FLAT': 'text-warning',
    }
    return classes.get(value, 'text-muted')


@register.filter
def confidence_badge(value):
    if value == 'HIGH':
        return '<span class="badge bg-success">HIGH ✅</span>'
    return '<span class="badge bg-warning text-dark">LOW ⚠️</span>'


@register.filter
def format_inr(value):
    try:
        val = float(value)
        if val >= 10000000:
            return f'₹{val/10000000:.2f} Cr'
        elif val >= 100000:
            return f'₹{val/100000:.2f} L'
        else:
            return f'₹{val:,.2f}'
    except (ValueError, TypeError):
        return value


@register.filter
def pct_color(value):
    try:
        val = float(value)
        if val > 0:
            return 'color: #00C853'
        elif val < 0:
            return 'color: #FF1744'
        return 'color: #FFC107'
    except (ValueError, TypeError):
        return ''


@register.filter
def multiply(value, arg):
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0
