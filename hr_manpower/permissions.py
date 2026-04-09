from rest_framework.permissions import BasePermission

# Group names — create these in Django Admin once
HR_ADMIN = 'HR Admin'
HR_OFFICER = 'HR Officer'
PROJECT_MANAGER = 'Project Manager'
FOREMAN = 'Foreman'
FINANCE = 'Finance'
ADMIN = 'Admin'


def _in_group(user, group_name):
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    return user.groups.filter(name=group_name).exists()


def _in_any_group(user, *group_names):
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    return user.groups.filter(name__in=group_names).exists()


class IsHRAdmin(BasePermission):
    """Full HR control: Employee master, onboarding, camp management."""
    message = 'Requires HR Admin role.'

    def has_permission(self, request, view):
        return _in_any_group(request.user, HR_ADMIN, ADMIN)


class IsHROfficer(BasePermission):
    """Attendance review and input."""
    message = 'Requires HR Officer role or above.'

    def has_permission(self, request, view):
        return _in_any_group(request.user, HR_ADMIN, HR_OFFICER, ADMIN)


class IsProjectManager(BasePermission):
    """Approve attendance, approve project assignments."""
    message = 'Requires Project Manager role or above.'

    def has_permission(self, request, view):
        return _in_any_group(request.user, HR_ADMIN, PROJECT_MANAGER, ADMIN)


class IsForeman(BasePermission):
    """Enter attendance in Draft status."""
    message = 'Requires Foreman role or above.'

    def has_permission(self, request, view):
        return _in_any_group(request.user, HR_ADMIN, HR_OFFICER, PROJECT_MANAGER, FOREMAN, ADMIN)


class IsFinance(BasePermission):
    """View labor costs and cost reports."""
    message = 'Requires Finance role or above.'

    def has_permission(self, request, view):
        return _in_any_group(request.user, HR_ADMIN, FINANCE, ADMIN)


class IsHRAdminOrReadOnly(BasePermission):
    """HR Admin for write ops; any authenticated user for reads."""
    message = 'Write access requires HR Admin role.'

    def has_permission(self, request, view):
        if request.method in ('GET', 'HEAD', 'OPTIONS'):
            return request.user and request.user.is_authenticated
        return _in_any_group(request.user, HR_ADMIN, ADMIN)
