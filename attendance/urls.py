from django.urls import path
from .views import *

urlpatterns = [
    # API endpoints
    path('register-user/', RegisterUserView.as_view(), name='register-user'),
    path('mark-attendance/', MarkAttendanceView.as_view(), name='mark-attendance'),
    path('admin/login/', AdminLoginView.as_view(), name='admin-login'),
    path('stats/', AttendanceStatsView.as_view(), name='attendance-stats'),
    path('alerts/', AttendanceAlertsView.as_view(), name='attendance-alerts'),
    path('employees/', EmployeeListView.as_view(), name='employee-list'),
    path('employees/import/', ImportEmployeesView.as_view(), name='import-employees'),
    path('employees/template/', DownloadEmployeeTemplateView.as_view(), name='download-employee-template'),
    path('employees/bulk-edit/template/', BulkEditTemplateView.as_view(), name='bulk-edit-template'),
    path('employees/bulk-edit/', BulkEditEmployeesView.as_view(), name='bulk-edit-employees'),
    path('employees/add/', AdminAddEmployeeView.as_view(), name='admin-add-employee'),
    path('employees/edit/<int:employee_id>/', AdminEditEmployeeView.as_view(), name='admin-edit-employee'),
    path('employees/<int:employee_id>/status-history/', EmployeeStatusHistoryView.as_view(), name='admin-employee-status-history'),
    path('employees/<int:employee_id>/site-history/', EmployeeSiteHistoryView.as_view(), name='admin-employee-site-history'),
    path('employees/<int:employee_id>/attachments/', EmployeeAttachmentsView.as_view(), name='admin-employee-attachments'),
    path('employees/<int:employee_id>/sick-leave/', EmployeeSickLeaveView.as_view(), name='admin-employee-sick-leave'),
    path('employees/<int:employee_id>/mark-day/', MarkDayView.as_view(), name='admin-employee-mark-day'),
    path('employees/delete/<int:employee_id>/', AdminDeleteEmployeeView.as_view(), name='admin-delete-employee'),
    path('employees/delete/bulk/', AdminBulkDeleteEmployeeView.as_view(), name='admin-bulk-delete-employees'),
    path('employees/export/', ExportAttendanceView.as_view(), name='export-attendance'),
    path('employees/export-filtered/', ExportEmployeesView.as_view(), name='export-filtered-employees'),
    path('sites/', SiteListView.as_view(), name='site-list'),
    
    # Admin Dashboard Template Views
    path('dashboard/login/', admin_login_view, name='admin-login'),
    path('dashboard/', admin_dashboard_view, name='admin-dashboard'),
    path('dashboard/user-face/', admin_user_face_view, name='admin-user-face'),
    path('dashboard/user-face/export/', ExportFaceEnrollmentView.as_view(), name='export-face-enrollment'),
    path('dashboard/user/<int:user_id>/', admin_user_detail_view, name='admin-user-detail'),
    path('dashboard/logout/', admin_logout_view, name='admin-logout'),
    path('dashboard/downloads/', admin_downloads_view, name='admin-downloads'),
    
    # Sites Management
    path('dashboard/sites/', admin_sites_view, name='admin-sites'),
    path('dashboard/sites/add/', admin_add_site, name='admin-add-site'),
    path('dashboard/sites/edit/<int:site_id>/', admin_edit_site, name='admin-edit-site'),
    path('dashboard/sites/delete/<int:site_id>/', admin_delete_site, name='admin-delete-site'),
    path('dashboard/sites/delete/bulk/', AdminBulkDeleteSiteView.as_view(), name='admin-bulk-delete-sites'),
    path('sites/import/', ImportSitesView.as_view(), name='import-sites'),
    path('sites/import-schedule/', ImportSitesScheduleView.as_view(), name='import-site-schedule'),
    path('sites/schedule-template/', DownloadSiteScheduleTemplateView.as_view(), name='download-site-schedule-template'),
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
    path('dashboard/salary-report/', AdminSalaryReportView.as_view(), name='admin-salary-report'),
    path('dashboard/download-salary-slip/<int:employee_id>/<int:month>/<int:year>/', DownloadSalarySlipView.as_view(), name='download-salary-slip'),
    path('dashboard/reports/export/', export_reports_view, name='export-reports'),
    path('dashboard/monthly-report/', monthly_report_view, name='monthly-report'),
    path('dashboard/monthly-report/export/', export_monthly_report, name='export-monthly-report'),
    path('dashboard/monthly-report/export/', export_monthly_report, name='export-monthly-report'),
    
    # Employee History API (Mobile)
    path('api/employees/<int:employee_id>/history/', EmployeeAttendanceHistoryView.as_view(), name='employee-attendance-history'),
    
    # Reports API
    path('api/reports/data/', AttendanceReportDataView.as_view(), name='api-reports-data'),
]

