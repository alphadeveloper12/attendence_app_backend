from django.urls import path
from .views import *

urlpatterns = [
    # API endpoints
    path('register-user/', RegisterUserView.as_view(), name='register-user'),
    path('mark-attendance/', MarkAttendanceView.as_view(), name='mark-attendance'),
    path('admin/login/', AdminLoginView.as_view(), name='admin-login'),
    path('stats/', AttendanceStatsView.as_view(), name='attendance-stats'),
    path('employees/', EmployeeListView.as_view(), name='employee-list'),
    path('employees/import/', ImportEmployeesView.as_view(), name='import-employees'),
    path('employees/add/', AdminAddEmployeeView.as_view(), name='admin-add-employee'),
    path('employees/edit/<int:employee_id>/', AdminEditEmployeeView.as_view(), name='admin-edit-employee'),
    path('employees/delete/<int:employee_id>/', AdminDeleteEmployeeView.as_view(), name='admin-delete-employee'),
    path('employees/delete/bulk/', AdminBulkDeleteEmployeeView.as_view(), name='admin-bulk-delete-employees'),
    path('employees/export/', ExportAttendanceView.as_view(), name='export-attendance'),
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
    path('dashboard/sites/delete/bulk/', AdminBulkDeleteSiteView.as_view(), name='admin-bulk-delete-sites'),
    path('sites/import/', ImportSitesView.as_view(), name='import-sites'),
    path('dashboard/sites/<int:site_id>/', admin_site_detail_view, name='admin-site-detail'),
    
    # API endpoint for site coordinates
    path('api/sites/<int:site_id>/coordinates/', SiteCoordinatesView.as_view(), name='site-coordinates-api'),
    
    # Site Admin Management
    path('dashboard/site-admins/', admin_site_admins_view, name='admin-site-admins'),
    path('dashboard/site-admins/add/', admin_add_site_admin, name='admin-add-site-admin'),
    path('dashboard/site-admins/edit/<int:admin_id>/', admin_edit_site_admin, name='admin-edit-site-admin'),
    path('dashboard/site-admins/delete/<int:admin_id>/', admin_delete_site_admin, name='admin-delete-site-admin'),
    path('dashboard/site-admins/delete/bulk/', admin_bulk_delete_site_admins, name='admin-bulk-delete-site-admins'),
    
    # App Builds
    path('builds/upload/', UploadBuildView.as_view(), name='upload-build'),
    path('builds/download/<str:app_type>/', DownloadBuildView.as_view(), name='download-build'),

    # Reports
    path('dashboard/reports/', admin_reports_view, name='admin-reports'),
    path('dashboard/reports/export/', export_reports_view, name='export-reports'),
]

