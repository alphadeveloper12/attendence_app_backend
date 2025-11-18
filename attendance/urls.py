from django.urls import path
from .views import *

urlpatterns = [
    path('register-user/', RegisterUserView.as_view(), name='register-user'),
    path('mark-attendance/', MarkAttendanceView.as_view(), name='mark-attendance'),
    # path('admin/manage-users/', AdminManageUsersView.as_view(), name='manage-users'),
    # path('admin/attendance/', AdminViewAttendanceView.as_view(), name='view-attendance'),
    path('admin/login/', AdminLoginView.as_view(), name='admin-login'),
    path('stats/', AttendanceStatsView.as_view(), name='attendance-stats'),
    path('employees/', EmployeeListView.as_view(), name='employee-list'),


]
