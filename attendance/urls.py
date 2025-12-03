from django.urls import path
from .views import *

urlpatterns = [
    # API endpoints
    path('register-user/', RegisterUserView.as_view(), name='register-user'),
    path('mark-attendance/', MarkAttendanceView.as_view(), name='mark-attendance'),
    path('admin/login/', AdminLoginView.as_view(), name='admin-login'),
    path('stats/', AttendanceStatsView.as_view(), name='attendance-stats'),
    path('employees/', EmployeeListView.as_view(), name='employee-list'),
    path('departments/', DepartmentListView.as_view(), name='department-list'),
    path('sites/', SiteListView.as_view(), name='site-list'),
    
    # Admin Dashboard Template Views
    path('dashboard/login/', admin_login_view, name='admin-login'),
    path('dashboard/', admin_dashboard_view, name='admin-dashboard'),
    path('dashboard/user/<int:user_id>/', admin_user_detail_view, name='admin-user-detail'),
    path('dashboard/logout/', admin_logout_view, name='admin-logout'),
    
    # Sites Management
    path('dashboard/sites/', admin_sites_view, name='admin-sites'),
    path('dashboard/sites/add/', admin_add_site, name='admin-add-site'),
    path('dashboard/sites/edit/<int:site_id>/', admin_edit_site, name='admin-edit-site'),
    path('dashboard/sites/delete/<int:site_id>/', admin_delete_site, name='admin-delete-site'),
    
    # Departments Management
    path('dashboard/departments/', admin_departments_view, name='admin-departments'),
    path('dashboard/departments/add/', admin_add_department, name='admin-add-department'),
    path('dashboard/departments/edit/<int:dept_id>/', admin_edit_department, name='admin-edit-department'),
    path('dashboard/departments/delete/<int:dept_id>/', admin_delete_department, name='admin-delete-department'),
]


