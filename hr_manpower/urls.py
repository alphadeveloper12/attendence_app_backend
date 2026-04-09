from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TradeViewSet,
    EmployeeHRProfileViewSet,
    EmployeeWithHRProfileView,
    ProjectViewSet,
    ActivityViewSet,
    ManpowerDemandViewSet,
    ProjectAssignmentViewSet,
    AttendanceExtensionViewSet,
    ActualCostViewSet,
    ProductivityRecordViewSet,
    CampViewSet,
    CampRoomViewSet,
    CampAllocationViewSet,
    OnboardingView,
    OffboardingView,
    ProjectTransferView,
)

router = DefaultRouter()
router.register(r'trades', TradeViewSet, basename='trade')
router.register(r'hr-profiles', EmployeeHRProfileViewSet, basename='hr-profile')
router.register(r'projects', ProjectViewSet, basename='project')
router.register(r'activities', ActivityViewSet, basename='activity')
router.register(r'manpower-demands', ManpowerDemandViewSet, basename='manpower-demand')
router.register(r'assignments', ProjectAssignmentViewSet, basename='assignment')
router.register(r'attendance-extensions', AttendanceExtensionViewSet, basename='attendance-extension')
router.register(r'actual-costs', ActualCostViewSet, basename='actual-cost')
router.register(r'productivity', ProductivityRecordViewSet, basename='productivity')
router.register(r'camps', CampViewSet, basename='camp')
router.register(r'camp-rooms', CampRoomViewSet, basename='camp-room')
router.register(r'camp-allocations', CampAllocationViewSet, basename='camp-allocation')

urlpatterns = [
    path('', include(router.urls)),

    # Employees enriched with HR profile (read-only)
    path('employees/hr/', EmployeeWithHRProfileView.as_view(), name='employees-with-hr-profile'),

    # Workflow endpoints
    path('onboarding/', OnboardingView.as_view(), name='onboarding'),
    path('offboarding/', OffboardingView.as_view(), name='offboarding'),
    path('transfer/', ProjectTransferView.as_view(), name='project-transfer'),
]
