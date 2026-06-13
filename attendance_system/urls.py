
from django.urls import reverse
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.shortcuts import redirect, render


def root_landing(request):
    """Marketing landing page at /.

    Use `?stay=1` (e.g. /?stay=1) to force-show the landing page even while
    logged in — useful for previewing the marketing site.
    Without that flag, logged-in staff get sent straight to their dashboard.
    """
    if (request.user.is_authenticated and request.user.is_staff
            and not request.GET.get('stay')):
        return redirect(reverse('admin-dashboard'))
    return render(request, 'landing.html')


def legacy_dashboard_redirect(request, slug=''):
    """Catches old bookmarks like /api/attendance/dashboard/login/ and
    redirects them to the clean URLs.

    Special cases:
      /api/attendance/dashboard/login/   → /login/   (auth moved out of /dashboard/)
      /api/attendance/dashboard/logout/  → /logout/  (same)
      /api/attendance/dashboard/<rest>   → /dashboard/<rest>
    """
    slug = (slug or '').lstrip('/')
    if slug in ('login/', 'login'):    target = '/login/'
    elif slug in ('logout/', 'logout'): target = '/logout/'
    elif slug:                          target = '/dashboard/' + slug
    else:                                target = '/dashboard/'
    qs = request.META.get('QUERY_STRING', '')
    if qs:
        target += '?' + qs
    return redirect(target, permanent=False)


urlpatterns = [
    # /                  → marketing landing page
    path('', root_landing, name='landing'),

    # Django admin
    path('admin/', admin.site.urls),

    # JSON API + backward-compat for old URLs.
    # The mobile app + every existing fetch() in templates uses these.
    # Registered BEFORE the clean web mount so that `reverse('admin-login')`
    # picks the LATER (clean) registration — Django's name lookup returns
    # the LAST registered URL pattern, so order here matters.
    path('api/attendance/', include('attendance.urls')),
    path('api/hr/',         include('hr_manpower.urls')),

    # Friendly redirects for old browser bookmarks.
    # /api/attendance/dashboard/login/   → /login/
    # /api/attendance/dashboard/logout/  → /logout/
    # /api/attendance/dashboard/<rest>   → /dashboard/<rest>
    # /api/attendance/login/             → /login/
    # /api/attendance/logout/            → /logout/
    path('api/attendance/dashboard/<path:slug>', lambda r, slug: legacy_dashboard_redirect(r, slug)),
    path('api/attendance/dashboard/',             lambda r: legacy_dashboard_redirect(r)),
    path('api/attendance/login/',                 lambda r: legacy_dashboard_redirect(r, 'login/')),
    path('api/attendance/logout/',                lambda r: legacy_dashboard_redirect(r, 'logout/')),

    # Clean web URLs:  /login/, /dashboard/, /dashboard/sites/, etc.
    # Registered LAST so `reverse('admin-login')` returns /login/.
    path('', include('attendance.web_urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
