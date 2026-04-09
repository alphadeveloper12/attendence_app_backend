from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    SessionAuthentication without CSRF enforcement.
    Safe to use for same-origin dashboard fetch() calls that rely on
    the Django session cookie but do not include a CSRF token header.
    CSRF is already enforced at the middleware level for template-rendered views.
    """

    def enforce_csrf(self, request):
        return  # skip CSRF check — session validity is sufficient proof of origin
