"""URL configuration split into two logical halves.

`web_urlpatterns` — user-facing template pages. Mounted at root `/` by the
project urls.py so URLs are clean: `/dashboard/`, `/login/`, `/dashboard/sites/`
etc. instead of `/api/attendance/dashboard/`.

`api_urlpatterns` — JSON endpoints used by the mobile app + AJAX calls from
the admin templates. Mounted at `/api/attendance/` (kept for backward compat
with the mobile app and any existing fetch() URLs in the templates).

`urlpatterns` — the union; included at `/api/attendance/` so every old URL
still resolves. New code should reverse() by name; both prefixes resolve to
the SAME view, but `reverse()` picks the FIRST registered prefix, which is
the clean web one.
"""
from django.urls import path
from .views import *


# ── Web pages (template views — render HTML) ──────────────────────────────
web_urlpatterns = [
    # Auth
    path('login/',  admin_login_view,  name='admin-login'),
    path('logout/', admin_logout_view, name='admin-logout'),

    # Dashboard root + main pages
    path('dashboard/',                        admin_dashboard_view,             name='admin-dashboard'),
    path('dashboard/user-face/',              admin_user_face_view,             name='admin-user-face'),
    path('dashboard/user/<int:user_id>/',     admin_user_detail_view,           name='admin-user-detail'),
    path('dashboard/downloads/',              admin_downloads_view,             name='admin-downloads'),
    path('dashboard/reports/',                admin_reports_view,               name='admin-reports'),
    path('dashboard/salary-report/',          AdminSalaryReportView.as_view(),  name='admin-salary-report'),
    path('dashboard/monthly-report/',         monthly_report_view,              name='monthly-report'),
    path('dashboard/distribution-list/',      admin_distribution_list_view,     name='admin-distribution-list'),
    path('dashboard/departments-management/', admin_departments_management_view, name='admin-departments-management'),
    path('dashboard/attrition-risk/',         admin_attrition_risk_view,        name='admin-attrition-risk'),
    path('dashboard/document-expiry/',        admin_document_expiry_view,       name='admin-document-expiry'),
    path('dashboard/manpower-recommendations/', admin_manpower_recommendations_view, name='admin-manpower-recommendations'),
    path('dashboard/ask-data/',               admin_ask_data_view,              name='admin-ask-data'),
    path('dashboard/geofence-tuning/',        admin_geofence_tuning_view,       name='admin-geofence-tuning'),
    path('dashboard/settings/',               admin_settings_view,              name='admin-settings'),

    # Sites management (template views)
    path('dashboard/sites/',                      admin_sites_view,        name='admin-sites'),
    path('dashboard/sites/add/',                  admin_add_site,          name='admin-add-site'),
    path('dashboard/sites/edit/<int:site_id>/',   admin_edit_site,         name='admin-edit-site'),
    path('dashboard/sites/delete/<int:site_id>/', admin_delete_site,       name='admin-delete-site'),
    path('dashboard/sites/<int:site_id>/',        admin_site_detail_view,  name='admin-site-detail'),

    # Site admin management (template views)
    path('dashboard/site-admins/',                       admin_site_admins_view,      name='admin-site-admins'),
    path('dashboard/site-admins/add/',                   admin_add_site_admin,        name='admin-add-site-admin'),
    path('dashboard/site-admins/edit/<int:admin_id>/',   admin_edit_site_admin,       name='admin-edit-site-admin'),
    path('dashboard/site-admins/delete/<int:admin_id>/', admin_delete_site_admin,     name='admin-delete-site-admin'),
]


# ── JSON API endpoints (used by mobile app + AJAX in templates) ───────────
api_urlpatterns = [
    # Auth + identity
    path('register-user/',       RegisterUserView.as_view(),       name='register-user'),
    path('mark-attendance/',     MarkAttendanceView.as_view(),     name='mark-attendance'),
    path('admin/login/',         AdminLoginView.as_view(),         name='admin-login-api'),

    # Stats + alerts
    path('stats/',                AttendanceStatsView.as_view(),         name='attendance-stats'),
    path('alerts/',               AttendanceAlertsView.as_view(),        name='attendance-alerts'),
    path('alerts/export/',        AttendanceAlertsExportView.as_view(),  name='attendance-alerts-export'),

    # Employees CRUD + import/export
    path('employees/',                                EmployeeListView.as_view(),               name='employee-list'),
    path('employees/import/',                         ImportEmployeesView.as_view(),            name='import-employees'),
    path('employees/import-preview/',                 ExcelMappingPreviewView.as_view(),        name='employees-import-preview'),
    path('employees/template/',                       DownloadEmployeeTemplateView.as_view(),   name='download-employee-template'),
    path('employees/bulk-edit/template/',             BulkEditTemplateView.as_view(),           name='bulk-edit-template'),
    path('employees/bulk-edit/',                      BulkEditEmployeesView.as_view(),          name='bulk-edit-employees'),
    path('employees/add/',                            AdminAddEmployeeView.as_view(),           name='admin-add-employee'),
    path('employees/edit/<int:employee_id>/',         AdminEditEmployeeView.as_view(),          name='admin-edit-employee'),
    path('employees/delete/<int:employee_id>/',       AdminDeleteEmployeeView.as_view(),        name='admin-delete-employee'),
    path('employees/delete/bulk/',                    AdminBulkDeleteEmployeeView.as_view(),    name='admin-bulk-delete-employees'),
    path('employees/export/',                         ExportAttendanceView.as_view(),           name='export-attendance'),
    path('employees/export-filtered/',                ExportEmployeesView.as_view(),            name='export-filtered-employees'),
    path('employees/export-selected/',                ExportSelectedEmployeesView.as_view(),    name='export-selected-employees'),
    path('employees/<int:employee_id>/status-history/',  EmployeeStatusHistoryView.as_view(),   name='admin-employee-status-history'),
    path('employees/<int:employee_id>/site-history/',    EmployeeSiteHistoryView.as_view(),     name='admin-employee-site-history'),
    path('employees/<int:employee_id>/attachments/',     EmployeeAttachmentsView.as_view(),     name='admin-employee-attachments'),
    path('employees/<int:employee_id>/salary-history/',  EmployeeSalaryHistoryView.as_view(),   name='admin-employee-salary-history'),
    path('employees/<int:employee_id>/sick-leave/',      EmployeeSickLeaveView.as_view(),       name='admin-employee-sick-leave'),
    path('employees/<int:employee_id>/mark-day/',        MarkDayView.as_view(),                 name='admin-employee-mark-day'),

    # Job categories + departments (catalogue)
    path('job-categories/',                          JobCategoryListView.as_view(),         name='job-categories'),
    path('departments/',                             DepartmentListView.as_view(),          name='departments'),
    path('admin-departments/',                       AdminDepartmentsView.as_view(),        name='admin-departments'),
    path('admin-departments/<int:dept_id>/',         AdminDepartmentDetailView.as_view(),   name='admin-department-detail'),
    path('admin-positions/',                         AdminPositionsView.as_view(),          name='admin-positions'),
    path('admin-positions/<int:pos_id>/',            AdminPositionDetailView.as_view(),     name='admin-position-detail'),

    # Settings (global app configuration — superuser only)
    path('settings/',                                AppSettingsView.as_view(),             name='app-settings'),
    path('settings/holidays/',                       PublicHolidayView.as_view(),           name='public-holidays'),
    path('settings/holidays/<int:holiday_id>/',      PublicHolidayView.as_view(),           name='public-holiday-detail'),

    # Analytics
    path('analytics/attrition-risk/',                AttritionRiskView.as_view(),           name='analytics-attrition-risk'),
    path('analytics/document-expiry/',               DocumentExpiryView.as_view(),          name='analytics-document-expiry'),
    path('analytics/manpower-recommendations/',      ManpowerRecommendationsView.as_view(), name='analytics-manpower-recommendations'),
    path('analytics/executive-summary/',             ExecutiveSummaryView.as_view(),        name='analytics-executive-summary'),
    path('analytics/ask/',                           AskDataView.as_view(),                 name='analytics-ask-data'),
    path('analytics/geofence-tuning/',               GeofenceTuningView.as_view(),          name='analytics-geofence-tuning'),
    path('analytics/site-activity/',                 SiteActivityView.as_view(),            name='analytics-site-activity'),

    # Distribution list
    path('distribution/',                            ManpowerDistributionView.as_view(),       name='manpower-distribution'),
    path('distribution/export/',                     ManpowerDistributionExportView.as_view(), name='manpower-distribution-export'),

    # Sites — JSON + import (template paths are in web_urlpatterns)
    path('sites/',                                   SiteListView.as_view(),                    name='site-list'),
    path('sites/import/',                            ImportSitesView.as_view(),                 name='import-sites'),
    path('sites/import-schedule/',                   ImportSitesScheduleView.as_view(),         name='import-site-schedule'),
    path('sites/schedule-template/',                 DownloadSiteScheduleTemplateView.as_view(),name='download-site-schedule-template'),
    path('dashboard/sites/delete/bulk/',             AdminBulkDeleteSiteView.as_view(),         name='admin-bulk-delete-sites'),
    path('api/sites/<int:site_id>/coordinates/',     SiteCoordinatesView.as_view(),             name='site-coordinates-api'),

    # Face enrollment + user-face export
    path('dashboard/user-face/export/',              ExportFaceEnrollmentView.as_view(),       name='export-face-enrollment'),

    # Reports
    path('dashboard/download-salary-slip/<int:employee_id>/<int:month>/<int:year>/', DownloadSalarySlipView.as_view(), name='download-salary-slip'),
    path('dashboard/reports/export/',                export_reports_view,                       name='export-reports'),
    path('dashboard/monthly-report/export/',         export_monthly_report,                     name='export-monthly-report'),
    path('dashboard/monthly-report/exec-summary/',   monthly_report_exec_summary,               name='monthly-report-exec-summary'),
    path('dashboard/monthly-report/calendar/<int:employee_id>/', monthly_report_employee_calendar, name='monthly-report-employee-calendar'),

    # Site admin bulk delete (JSON)
    path('dashboard/site-admins/delete/bulk/',       admin_bulk_delete_site_admins,             name='admin-bulk-delete-site-admins'),

    # App builds
    path('builds/upload/',                           UploadBuildView.as_view(),                 name='upload-build'),
    path('builds/download/<str:app_type>/',          DownloadBuildView.as_view(),               name='download-build'),

    # Employee history + reports data (mobile)
    path('api/employees/<int:employee_id>/history/', EmployeeAttendanceHistoryView.as_view(),  name='employee-attendance-history'),
    path('api/reports/data/',                        AttendanceReportDataView.as_view(),        name='api-reports-data'),
    path('api/reports/exec-summary/',                daily_report_exec_summary,                 name='api-reports-exec-summary'),
]


# Backward-compat: keep all paths reachable under /api/attendance/ too.
# Mobile clients + existing fetch() calls in dashboard.html use these URLs.
urlpatterns = web_urlpatterns + api_urlpatterns

