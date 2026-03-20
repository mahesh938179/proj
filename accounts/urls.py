from django.urls import path
from django.views.generic import TemplateView
from . import views

app_name = 'accounts'

urlpatterns = [
    # PWA Support
    path('manifest.json', TemplateView.as_view(template_name="pwa/manifest.json", content_type='application/json'), name='manifest'),
    path('service-worker.js', TemplateView.as_view(template_name="pwa/service-worker.js", content_type='application/javascript'), name='service_worker'),

    # Auth
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # User Profile
    path('profile/', views.profile_view, name='profile'),
    path('password-change/', views.change_password_view, name='password_change'),
    path('delete-account/', views.delete_account_view, name='delete_account'),
    
    # Admin Management
    path('admin/dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin/user/create/', views.user_create_view, name='user_create'),
    path('admin/user/<int:user_id>/update/', views.admin_user_update_view, name='admin_user_update'),
    path('admin/user/<int:user_id>/password/', views.admin_user_password_change_view, name='admin_user_password_reset'),
    path('admin/user/<int:user_id>/delete/', views.user_delete_view, name='user_delete'),
    path('admin/user/<int:user_id>/role/', views.user_update_role_view, name='user_update_role'),
    
    # Deletion Requests
    path('admin/deletion-requests/', views.deletion_requests_list_view, name='deletion_requests'),
    path('admin/deletion-requests/<int:request_id>/approve/', views.approve_deletion_view, name='approve_deletion'),
    path('admin/deletion-requests/<int:request_id>/reject/', views.reject_deletion_view, name='reject_deletion'),
    path('dismiss-rejection/', views.dismiss_rejection_view, name='dismiss_rejection'),
]
