
from django.urls import reverse
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.shortcuts import redirect

def root_redirect(request):
    if request.user.is_authenticated:
        return redirect(reverse('admin-dashboard'))
    return redirect(reverse('admin-login'))  # direct to login instead of dashboard

urlpatterns = [
    path('', root_redirect),
    path('admin/', admin.site.urls),
    path('api/attendance/', include('attendance.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
