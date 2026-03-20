from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q
from django.contrib.auth.models import User
from .forms import RegisterForm, LoginForm, ProfileUpdateForm, UserPasswordChangeForm
from .models import Role, Profile
from .decorators import admin_required, superadmin_required

@admin_required
def admin_dashboard_view(request):
    """Admin dashboard overview"""
    current_user_role = request.user.profile.role
    
    if current_user_role == Role.SUPERADMIN:
        users = User.objects.all().select_related('profile').order_by('-date_joined')
    else:
        # Admins can only see regular users
        users = User.objects.filter(profile__role=Role.USER).select_related('profile').order_by('-date_joined')
    
    # Filter search
    query = request.GET.get('q', '')
    if query:
        users = users.filter(
            Q(username__icontains=query) | Q(email__icontains=query)
        )
    
    total_users = User.objects.count()
    if current_user_role == Role.SUPERADMIN:
        admins_count = Profile.objects.filter(role=Role.ADMIN).count()
        superadmins_count = Profile.objects.filter(role=Role.SUPERADMIN).count()
    else:
        admins_count = 0
        superadmins_count = 0
    
    context = {
        'users': users,
        'total_users': total_users,
        'admins_count': admins_count,
        'superadmins_count': superadmins_count,
        'search_query': query,
    }
    return render(request, 'accounts/admin/dashboard.html', context)

@admin_required
def user_create_view(request):
    """View to create regular users/admins (admins can only create regular users, superadmins can create admins)"""
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Default to USER unless specified and allowed
            user_role = request.POST.get('role', Role.USER)
            
            # Check permissions: Admins can't create admins/superadmins
            can_assign = False
            if user_role == Role.USER:
                can_assign = True
            elif user_role == Role.ADMIN and request.user.profile.is_superadmin():
                can_assign = True
                
            if can_assign:
                user.save()
                user.profile.role = user_role
                user.profile.save()
                messages.success(request, f"User {user.username} created successfully!")
                return redirect('accounts:admin_dashboard')
            else:
                messages.error(request, "Insufficient permission to assign this role.")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = RegisterForm()
    
    return render(request, 'accounts/admin/user_create.html', {
        'form': form,
        'roles': Role.choices
    })

@admin_required
def user_delete_view(request, user_id):
    """Delete a user, with permission checks"""
    target_user = User.objects.get(id=user_id)
    
    # Safety checks
    if target_user == request.user:
        messages.error(request, "You cannot delete yourself.")
    elif target_user.profile.is_superadmin() and not request.user.profile.is_superadmin():
        messages.error(request, "Admins cannot delete SuperAdmins.")
    elif target_user.profile.role == Role.ADMIN and not request.user.profile.is_superadmin():
        messages.error(request, "Only SuperAdmins can delete Admins.")
    else:
        target_user.delete()
        messages.success(request, f"User {target_user.username} has been deleted.")
        
    return redirect('accounts:admin_dashboard')

@admin_required
def user_update_role_view(request, user_id):
    """Update role for a user (SuperAdmin only can promote to Admin)"""
    if not request.user.profile.is_superadmin():
        messages.error(request, "Only SuperAdmins can manage roles.")
        return redirect('accounts:admin_dashboard')
        
    target_user = User.objects.get(id=user_id)
    
    if target_user == request.user:
        messages.error(request, "You cannot change your own role.")
        return redirect('accounts:admin_dashboard')
        
    new_role = request.POST.get('role')
    
    if new_role in Role.values:
        target_user.profile.role = new_role
        target_user.profile.save()
        messages.success(request, f"Role for {target_user.username} updated to {new_role}.")
    else:
        messages.error(request, "Invalid role specified.")
        
    return redirect('accounts:admin_dashboard')

@admin_required
def admin_user_update_view(request, user_id):
    """Admin view to update another user's profile details"""
    target_user = User.objects.get(id=user_id)
    current_role = request.user.profile.role
    
    # Permission checks: Admins can only edit regular Users
    if current_role == Role.ADMIN and target_user.profile.role != Role.USER:
        messages.error(request, "Permission denied. Admins can only edit regular Users.")
        return redirect('accounts:admin_dashboard')

    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=target_user)
        if form.is_valid():
            form.save()
            messages.success(request, f"Profile for {target_user.username} has been updated!")
            return redirect('accounts:admin_dashboard')
    else:
        form = ProfileUpdateForm(instance=target_user)
    
    return render(request, 'accounts/admin/user_update.html', {
        'form': form,
        'target_user': target_user
    })

@admin_required
def admin_user_password_change_view(request, user_id):
    """Admin view to change another user's password"""
    target_user = User.objects.get(id=user_id)
    current_role = request.user.profile.role
    
    # Permission checks: Admins can only edit regular Users
    if current_role == Role.ADMIN and target_user.profile.role != Role.USER:
        messages.error(request, "Permission denied. Admins can only edit regular Users.")
        return redirect('accounts:admin_dashboard')

    from django.contrib.auth.forms import SetPasswordForm
    if request.method == 'POST':
        form = SetPasswordForm(target_user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f"Password for {target_user.username} has been reset!")
            return redirect('accounts:admin_dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = SetPasswordForm(target_user)
    
    # Standardize classes for SetPasswordForm
    for field in form.fields:
        form.fields[field].widget.attrs['class'] = 'form-control'
        form.fields[field].widget.attrs['placeholder'] = form.fields[field].label

    return render(request, 'accounts/admin/user_password_reset.html', {
        'form': form,
        'target_user': target_user
    })

@login_required
def profile_view(request):
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile has been updated!")
            return redirect('accounts:profile')
    else:
        form = ProfileUpdateForm(instance=request.user)
    
    return render(request, 'accounts/profile.html', {'form': form})

@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = UserPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Keep the user logged in
            messages.success(request, "Your password has been changed successfully!")
            return redirect('accounts:profile')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = UserPasswordChangeForm(request.user)
    
    return render(request, 'accounts/change_password.html', {'form': form})

@login_required
def delete_account_view(request):
    """Users can only request deletion; SuperAdmin cannot delete themselves"""
    if request.user.profile.is_superadmin():
        messages.error(request, "SuperAdmins cannot delete their own accounts.")
        return redirect('accounts:profile')
        
    if request.method == 'POST':
        # Create a deletion request if it doesn't exist
        from .models import DeletionRequest
        DeletionRequest.objects.get_or_create(user=request.user)
        messages.warning(request, "Deletion request sent. An administrator must approve it.")
        return redirect('accounts:profile')
    return redirect('accounts:profile')

@admin_required
def deletion_requests_list_view(request):
    """Lists pending deletion requests (not yet rejected)"""
    from .models import DeletionRequest
    current_role = request.user.profile.role
    
    # Permission: Admins see only User level requests
    # Only show requests where is_rejected is False
    if current_role == Role.SUPERADMIN:
        pending_requests = DeletionRequest.objects.filter(is_rejected=False).select_related('user', 'user__profile')
    else:
        pending_requests = DeletionRequest.objects.filter(is_rejected=False, user__profile__role=Role.USER).select_related('user', 'user__profile')
        
    return render(request, 'accounts/admin/deletion_requests.html', {
        'requests': pending_requests
    })

@admin_required
def approve_deletion_view(request, request_id):
    """Approves a deletion request and deletes the user"""
    from .models import DeletionRequest
    del_request = DeletionRequest.objects.get(id=request_id)
    target_user = del_request.user
    current_role = request.user.profile.role
    
    can_approve = False
    # Regular Users can be approved by Admin or SuperAdmin
    if target_user.profile.role == Role.USER:
        can_approve = True
    # Admin Users can only be approved by SuperAdmin
    elif target_user.profile.role == Role.ADMIN and current_role == Role.SUPERADMIN:
        can_approve = True
        
    if can_approve:
        target_username = target_user.username
        target_user.delete()
        messages.success(request, f"User {target_username} has been permanently deleted.")
    else:
        messages.error(request, "Permission denied. Only SuperAdmins can approve Admin deletion.")
        
    return redirect('accounts:admin_dashboard')

@admin_required
def reject_deletion_view(request, request_id):
    """Rejects a deletion request by setting the flag"""
    from .models import DeletionRequest
    del_request = DeletionRequest.objects.get(id=request_id)
    target_username = del_request.user.username
    del_request.is_rejected = True
    del_request.save()
    messages.info(request, f"Deletion request for {target_username} has been rejected.")
    return redirect('accounts:deletion_requests')

@login_required
def dismiss_rejection_view(request):
    """Users can clear their rejected status to request again if needed"""
    from .models import DeletionRequest
    if hasattr(request.user, 'deletion_request') and request.user.deletion_request.is_rejected:
        request.user.deletion_request.delete()
        messages.info(request, "Rejection notice dismissed.")
    return redirect('accounts:profile')

def login_view(request):
    if request.user.is_authenticated:
        return redirect('predictions:dashboard')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('predictions:dashboard')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})

def register_view(request):
    if request.user.is_authenticated:
        return redirect('predictions:dashboard')
        
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful. Welcome to StockAI!")
            return redirect('predictions:dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = RegisterForm()
    
    return render(request, 'accounts/register.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('accounts:login')
