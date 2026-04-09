from django.contrib import admin
from .models import (
    Trade, EmployeeHRProfile, Project, Activity,
    ManpowerDemand, ProjectAssignment,
    AttendanceExtension, ActualCost,
    ProductivityRecord, Camp, CampRoom, CampAllocation,
)


@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ['trade_name', 'category', 'created_at']
    list_filter = ['category']
    search_fields = ['trade_name']


@admin.register(EmployeeHRProfile)
class EmployeeHRProfileAdmin(admin.ModelAdmin):
    list_display = ['employee_code', 'employee', 'trade', 'skill_level', 'employment_type', 'documents_valid']
    list_filter = ['skill_level', 'employment_type', 'documents_valid', 'trade']
    search_fields = ['employee_code', 'employee__name']
    raw_id_fields = ['employee', 'trade']


class ActivityInline(admin.TabularInline):
    model = Activity
    extra = 0
    fields = ['activity_code', 'activity_name', 'trade', 'start_date', 'end_date', 'status']


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['project_code', 'project_name', 'site', 'status', 'start_date', 'end_date']
    list_filter = ['status', 'site']
    search_fields = ['project_code', 'project_name']
    inlines = [ActivityInline]
    raw_id_fields = ['site', 'created_by']


@admin.register(Activity)
class ActivityAdmin(admin.ModelAdmin):
    list_display = ['activity_code', 'activity_name', 'project', 'trade', 'start_date', 'end_date', 'status']
    list_filter = ['status', 'trade']
    search_fields = ['activity_code', 'activity_name', 'project__project_code']
    raw_id_fields = ['project', 'trade']


@admin.register(ManpowerDemand)
class ManpowerDemandAdmin(admin.ModelAdmin):
    list_display = ['project', 'trade', 'required_qty', 'required_start_date', 'required_end_date', 'source']
    list_filter = ['source', 'trade']
    search_fields = ['project__project_code', 'trade__trade_name']
    raw_id_fields = ['project', 'activity', 'trade', 'created_by']


@admin.register(ProjectAssignment)
class ProjectAssignmentAdmin(admin.ModelAdmin):
    list_display = ['employee', 'project', 'activity', 'start_date', 'end_date', 'allocation_percent', 'status']
    list_filter = ['status']
    search_fields = ['employee__name', 'project__project_code']
    raw_id_fields = ['employee', 'project', 'activity', 'assigned_by']


@admin.register(AttendanceExtension)
class AttendanceExtensionAdmin(admin.ModelAdmin):
    list_display = ['attendance', 'project', 'activity', 'total_hours', 'source', 'approval_status', 'cost_posted']
    list_filter = ['approval_status', 'source', 'cost_posted']
    search_fields = ['attendance__user__name', 'project__project_code']
    raw_id_fields = ['attendance', 'project', 'activity', 'approved_by']


@admin.register(ActualCost)
class ActualCostAdmin(admin.ModelAdmin):
    list_display = ['employee', 'project', 'cost_date', 'regular_hours', 'overtime_hours', 'total_cost', 'cbs_code']
    list_filter = ['project', 'cbs_code']
    search_fields = ['employee__name', 'project__project_code']
    raw_id_fields = ['employee', 'project', 'activity', 'attendance_ext']
    readonly_fields = ['regular_cost', 'overtime_cost', 'total_cost', 'posted_at']


@admin.register(ProductivityRecord)
class ProductivityRecordAdmin(admin.ModelAdmin):
    list_display = ['project', 'activity', 'trade', 'record_date', 'man_hours_used', 'quantity_executed', 'productivity_rate']
    list_filter = ['trade', 'project']
    search_fields = ['project__project_code', 'activity__activity_name']
    raw_id_fields = ['project', 'activity', 'trade', 'entered_by']
    readonly_fields = ['productivity_rate']


class CampRoomInline(admin.TabularInline):
    model = CampRoom
    extra = 0
    fields = ['room_number', 'bed_count']


@admin.register(Camp)
class CampAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'capacity', 'is_active']
    list_filter = ['is_active']
    search_fields = ['name', 'location']
    inlines = [CampRoomInline]


@admin.register(CampRoom)
class CampRoomAdmin(admin.ModelAdmin):
    list_display = ['room_number', 'camp', 'bed_count']
    list_filter = ['camp']
    search_fields = ['room_number', 'camp__name']
    raw_id_fields = ['camp']


@admin.register(CampAllocation)
class CampAllocationAdmin(admin.ModelAdmin):
    list_display = ['employee', 'room', 'bed_number', 'start_date', 'end_date', 'status']
    list_filter = ['status', 'room__camp']
    search_fields = ['employee__name', 'room__room_number']
    raw_id_fields = ['employee', 'room']
