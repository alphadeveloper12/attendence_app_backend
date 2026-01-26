from rest_framework import permissions
from .models import AdminProfile

class IsSiteAdmin(permissions.BasePermission):
    """
    Allows access to site admins (users with an AdminProfile) or staff users.
    """
    def has_permission(self, request, view):
        if request.user and request.user.is_staff:
            return True
        # Check if user has an AdminProfile
        # Anonymous users have no profile
        if not request.user.is_authenticated:
            return False
            
        return AdminProfile.objects.filter(user=request.user).exists()
