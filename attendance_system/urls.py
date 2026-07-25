
from django.urls import reverse
from django.contrib import admin
from django.urls import path, include, re_path
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


def react_shell(request, rest=None):
    """Serves the React SPA shell. The bundle's client-side router (React Router)
    renders the right page based on the URL path. Used for all migrated pages
    while the legacy Django pages stay untouched.

    ``rest`` captures deep SPA paths (e.g. /app/employees/42) so a single shell
    view backs the whole client-side route tree.

    Passes the Google Maps key so the SPA's geofence maps can load it.
    """
    return render(request, 'react_base.html', {
        'google_maps_key': settings.GOOGLE_MAPS_API_KEY,
    })


# Backwards-compatible alias
react_home = react_shell


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
    # /                  → marketing landing page (legacy Django template)
    path('', root_landing, name='landing'),

    # NEW React pages (preview; the legacy Django pages stay untouched).
    # Convention: every migrated page mirrors its legacy path with a `-v2`
    # suffix — e.g. /dashboard/ → /dashboard-v2, /dashboard/reports/ →
    # /dashboard/reports-v2, /dashboard/sites/5/ → /dashboard/sites/5-v2.
    # A single catch-all serves the SPA shell for ANY url ending in `-v2` (with
    # optional trailing slash); React Router then renders the right page. This
    # never shadows the legacy pages, which never end in `-v2`.
    path('home-v2/', react_shell, name='home-v2'),
    path('login-v2/', react_shell, name='login-v2'),
    re_path(r'.*-v2/?$', react_shell, name='spa-v2'),

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
