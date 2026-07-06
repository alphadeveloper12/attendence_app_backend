"""Read-only enforcement.

A single choke point that blocks EVERY write request (POST/PUT/PATCH/DELETE)
for read-only "Viewer" admins, regardless of which endpoint they hit. This is
the authoritative guard — the hidden buttons in the UI are just cosmetics on top.
"""
from django.http import JsonResponse, HttpResponseForbidden

from .models import AdminProfile

SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


class ReadOnlyAdminMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method not in SAFE_METHODS:
            user = getattr(request, "user", None)
            if user is not None and user.is_authenticated and not user.is_superuser:
                is_viewer = AdminProfile.objects.filter(
                    user=user, role=AdminProfile.ROLE_VIEWER,
                ).exists()
                if is_viewer:
                    # Always allow logging out.
                    if not request.path.rstrip("/").endswith("logout"):
                        wants_json = (
                            request.path.startswith("/api/")
                            or request.headers.get("x-requested-with") == "XMLHttpRequest"
                            or "application/json" in request.headers.get("accept", "")
                        )
                        msg = "Read-only access — you do not have permission to add, edit, or delete."
                        if wants_json:
                            return JsonResponse({"error": msg}, status=403)
                        return HttpResponseForbidden(msg)
        return self.get_response(request)
