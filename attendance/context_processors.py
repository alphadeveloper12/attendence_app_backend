"""Template context processors.

`app_settings` injects the global AppSettings singleton and a resolved
per-request nav-visibility map into every rendered template, so base.html can
show/hide sidebar links and dashboard.html can read the salary/currency config.
"""
from .models import AppSettings, _default_nav_visibility


# Order here is informational; base.html owns the actual layout.
NAV_KEYS = list(_default_nav_visibility().keys())


def app_settings(request):
    # Be defensive: this runs on every render, including before the table
    # exists (initial migrate) or for anonymous users. Never raise.
    try:
        settings_obj = AppSettings.load()
    except Exception:
        return {
            'app_settings': None,
            'nav_visible': {k: True for k in NAV_KEYS},
            'currency_code': 'AED',
        }

    raw = settings_obj.nav_visibility or {}
    is_super = bool(getattr(getattr(request, 'user', None), 'is_superuser', False))

    # Superusers always see every link (they need access to manage them).
    # Non-superusers see a link only if it's enabled; missing keys default to
    # visible, except 'settings' which is superuser-only.
    nav_visible = {}
    for key in NAV_KEYS:
        if is_super:
            nav_visible[key] = True
        elif key == 'settings':
            nav_visible[key] = False
        else:
            nav_visible[key] = bool(raw.get(key, True))

    return {
        'app_settings': settings_obj,
        'nav_visible': nav_visible,
        'currency_code': settings_obj.currency_code or 'AED',
    }
