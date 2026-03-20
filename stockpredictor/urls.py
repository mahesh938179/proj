"""stockpredictor URL Configuration"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def health_check(request):
    """Health check endpoint for uptime monitoring / deployment checks."""
    return JsonResponse({
        'status': 'ok',
        'service': 'StockAI Predictor',
        'version': '2.0',
    })


@require_GET
def robots_txt(request):
    from django.http import HttpResponse
    content = "User-agent: *\nDisallow: /admin/\nDisallow: /api/\nAllow: /\n"
    return HttpResponse(content, content_type='text/plain')


from django.views.generic import TemplateView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health_check, name='health_check'),
    path('robots.txt', robots_txt, name='robots_txt'),
    
    # PWA Support
    path('manifest.json', TemplateView.as_view(
        template_name='pwa/manifest.json', 
        content_type='application/json'
    ), name='manifest_json'),
    path('service-worker.js', TemplateView.as_view(
        template_name='pwa/service-worker.js', 
        content_type='application/javascript'
    ), name='service_worker_js'),
    
    # Android TWA Verification
    path('.well-known/assetlinks.json', TemplateView.as_view(
        template_name='pwa/assetlinks.json', 
        content_type='application/json'
    ), name='assetlinks_json'),

    path('', include('predictions.urls')),
    path('accounts/', include('accounts.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
