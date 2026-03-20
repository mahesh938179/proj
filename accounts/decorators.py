from django.shortcuts import redirect
from django.contrib import messages
from functools import wraps

def role_required(roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('accounts:login')
            
            if request.user.profile.role in roles:
                return view_func(request, *args, **kwargs)
            
            messages.error(request, "You don't have permission to access this page.")
            return redirect('predictions:dashboard')
        return _wrapped_view
    return decorator

def superadmin_required(view_func):
    return role_required(['superadmin'])(view_func)

def admin_required(view_func):
    return role_required(['admin', 'superadmin'])(view_func)
