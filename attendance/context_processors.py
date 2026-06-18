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

    # Visibility toggles apply to EVERYONE, including super admins, so the
    # effect is visible immediately. Missing keys default to visible. The one
    # exception is 'settings': it stays visible to super admins no matter what,
    # so they can always get back here to re-enable links.
    nav_visible = {}
    for key in NAV_KEYS:
        if key == 'settings':
            nav_visible[key] = is_super
        else:
            nav_visible[key] = bool(raw.get(key, True))

    return {
        'app_settings': settings_obj,
        'nav_visible': nav_visible,
        'currency_code': settings_obj.currency_code or 'AED',
    }
