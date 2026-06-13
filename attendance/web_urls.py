"""Mountpoint for the user-facing template URLs at the project root.

Project `urls.py` does `path('', include('attendance.web_urls'))`, which
exposes /login/, /dashboard/, /dashboard/sites/, etc. — the clean URLs.

The actual list is defined in attendance.urls.web_urlpatterns; this module
just re-exports it under the standard name `urlpatterns` so Django's include()
can pick it up.
"""
from .urls import web_urlpatterns

urlpatterns = web_urlpatterns
