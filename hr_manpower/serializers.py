from rest_framework import serializers
from attendance.models import Employee, Attendance
from .models import (
    Trade, EmployeeHRProfile, Project, Activity,
    ManpowerDemand, ProjectAssignment,
    AttendanceExtension, ActualCost,
    ProductivityRecord, Camp, CampRoom, CampAllocation,
)


# ---------------------------------------------------------------------------
# TRADE
# ---------------------------------------------------------------------------

class TradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trade
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']


# ---------------------------------------------------------------------------
# EMPLOYEE HR PROFILE
# ---------------------------------------------------------------------------

class EmployeeHRProfileSerializer(serializers.ModelSerializer):
    employee_name = serializers.CharField(source='employee.name', read_only=True)
    trade_name = serializers.CharField(source='trade.trade_name', read_only=True)

    class Meta:
        model = EmployeeHRProfile
        fields = [
            'id', 'employee', 'employee_name', 'employee_code',
            'trade', 'trade_name', 'skill_level', 'employment_type',
            'basic_rate', 'overtime_rate_multiplier', 'cost_center_default',
            'documents_dms_ids', 'documents_valid',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class EmployeeWithHRProfileSerializer(serializers.ModelSerializer):
    """Read-only: combines base Employee fields with HR Profile fields."""
    hr_profile = EmployeeHRProfileSerializer(read_only=True)

    class Meta:
        model = Employee
        fields = [
            'id', 'name', 'email', 'phone', 'department', 'position',
            'nationality', 'status', 'date_of_joining', 'category',
            'site', 'sponsor', 'employer', 'hr_profile',
        ]


# ---------------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------------

class ActivitySerializer(serializers.ModelSerializer):
    trade_name = serializers.CharField(source='trade.trade_name', read_only=True)

    class Meta:
        model = Activity
        fields = [
            'id', 'project', 'activity_code', 'activity_name',
            'trade', 'trade_name', 'start_date', 'end_date',
            'planned_qty', 'qty_unit', 'status', 'created_at',
        ]
        read_only_fields = ['created_at']


class ProjectSerializer(serializers.ModelSerializer):
    activities = ActivitySerializer(many=True, read_only=True)
    site_name = serializers.CharField(source='site.name', read_only=True)

    class Meta:
        model = Project
        fields = [
            'id', 'project_code', 'project_name', 'site', 'site_name',
            'client_name', 'start_date', 'end_date', 'status', 'description',
            'created_by', 'created_at', 'updated_at', 'activities',
        ]
        read_only_fields = ['created_by', 'created_at', 'updated_at']


class ProjectListSerializer(serializers.ModelSerializer):
    """Lightweight list version without nested activities."""
    site_name = serializers.CharField(source='site.name', read_only=True)
    activity_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'project_code', 'project_name', 'site_name',
            'start_date', 'end_date', 'status', 'activity_count',
        ]

    def get_activity_count(self, obj):
        return obj.activities.count()


# ---------------------------------------------------------------------------
# MANPOWER DEMAND
# ---------------------------------------------------------------------------

class ManpowerDemandSerializer(serializers.ModelSerializer):
    trade_name = serializers.CharField(source='trade.trade_name', read_only=True)
    project_code = serializers.CharField(source='project.project_code', read_only=True)
    activity_name = serializers.CharField(source='activity.activity_name', read_only=True)
    # Supply: how many workers are currently assigned to this project+trade
    assigned_count = serializers.SerializerMethodField()
    gap = serializers.SerializerMethodField()

    class Meta:
        model = ManpowerDemand
        fields = [
            'id', 'project', 'project_code', 'activity', 'activity_name',
            'trade', 'trade_name', 'required_qty', 'required_start_date',
            'required_end_date', 'planned_man_hours', 'source', 'notes',
            'assigned_count', 'gap', 'created_at',
        ]
        read_only_fields = ['created_by', 'created_at', 'planned_man_hours']

    def get_assigned_count(self, obj):
        return ProjectAssignment.objects.filter(
            project=obj.project,
            employee__hr_profile__trade=obj.trade,
            status__in=['Planned', 'Active'],
            start_date__lte=obj.required_end_date,
        ).filter(
            models.Q(end_date__isnull=True) | models.Q(end_date__gte=obj.required_start_date)
        ).count()

    def get_gap(self, obj):
        return obj.required_qty - self.get_assigned_count(obj)


# ---------------------------------------------------------------------------
# PROJECT ASSIGNMENT
# ---------------------------------------------------------------------------

class ProjectAssignmentSerializer(serializers.ModelSerializer):
    employee_name = serializers.CharField(source='employee.name', read_only=True)
    project_code = serializers.CharField(source='project.project_code', read_only=True)
    activity_name = serializers.CharField(source='activity.activity_name', read_only=True)
    trade_name = serializers.SerializerMethodField()

    class Meta:
        model = ProjectAssignment
        fields = [
            'id', 'employee', 'employee_name', 'project', 'project_code',
            'activity', 'activity_name', 'trade_name',
            'start_date', 'end_date', 'allocation_percent', 'status',
            'assigned_by', 'released_at', 'notes', 'created_at', 'updated_at',
        ]
        read_only_fields = ['assigned_by', 'released_at', 'created_at', 'updated_at']

    def get_trade_name(self, obj):
        try:
            return obj.employee.hr_profile.trade.trade_name
        except Exception:
            return None

    def validate(self, data):
        employee = data.get('employee')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # On PATCH requests not all fields are sent — skip heavy validation
        if employee is None or start_date is None:
            return data

        # Overlap check: employee cannot be 100% assigned to 2 projects at the same time
        qs = ProjectAssignment.objects.filter(
            employee=employee,
            status__in=['Planned', 'Active'],
        ).exclude(pk=self.instance.pk if self.instance else None)

        if end_date:
            qs = qs.filter(start_date__lte=end_date)
        qs = qs.filter(
            models.Q(end_date__isnull=True) | models.Q(end_date__gte=start_date)
        )

        if qs.exists():
            raise serializers.ValidationError(
                'Employee already has an overlapping active assignment. '
                'Release the existing assignment first or adjust dates.'
            )

        # Document validity check
        try:
            if not employee.hr_profile.documents_valid:
                raise serializers.ValidationError(
                    'Employee mandatory documents are not valid/uploaded. '
                    'Update HR Profile documents before assigning to a project.'
                )
        except EmployeeHRProfile.DoesNotExist:
            raise serializers.ValidationError(
                'Employee does not have an HR Profile. Create one before assigning to a project.'
            )

        return data


# ---------------------------------------------------------------------------
# ATTENDANCE EXTENSION
# ---------------------------------------------------------------------------

class AttendanceExtensionSerializer(serializers.ModelSerializer):
    employee_name = serializers.SerializerMethodField()
    attendance_date = serializers.DateField(source='attendance.date', read_only=True)
    project_code = serializers.CharField(source='project.project_code', read_only=True)
    activity_name = serializers.CharField(source='activity.activity_name', read_only=True)
    approved_by_name = serializers.CharField(source='approved_by.username', read_only=True)

    class Meta:
        model = AttendanceExtension
        fields = [
            'id', 'attendance', 'employee_name', 'attendance_date',
            'project', 'project_code', 'activity', 'activity_name',
            'total_hours', 'source', 'approval_status',
            'approved_by', 'approved_by_name', 'approved_at',
            'rejection_reason', 'cost_posted', 'created_at', 'updated_at',
        ]
        read_only_fields = [
            'approved_by', 'approved_at', 'cost_posted', 'created_at', 'updated_at'
        ]

    def get_employee_name(self, obj):
        return obj.attendance.user.name


class AttendanceExtensionBulkCreateSerializer(serializers.Serializer):
    """For Foreman bulk daily entry: create AttendanceExtension records for a list of employees."""
    project = serializers.PrimaryKeyRelatedField(queryset=Project.objects.all())
    activity = serializers.PrimaryKeyRelatedField(
        queryset=Activity.objects.all(), required=False, allow_null=True
    )
    date = serializers.DateField()
    source = serializers.ChoiceField(choices=AttendanceExtension.SOURCE_CHOICES, default='manual')
    entries = serializers.ListField(
        child=serializers.DictField(), min_length=1,
        help_text='[{"attendance_id": 1, "total_hours": 8.0}, ...]'
    )


# ---------------------------------------------------------------------------
# ACTUAL COST
# ---------------------------------------------------------------------------

class ActualCostSerializer(serializers.ModelSerializer):
    employee_name = serializers.CharField(source='employee.name', read_only=True)
    project_code = serializers.CharField(source='project.project_code', read_only=True)
    activity_name = serializers.CharField(source='activity.activity_name', read_only=True)

    class Meta:
        model = ActualCost
        fields = [
            'id', 'project', 'project_code', 'activity', 'activity_name',
            'employee', 'employee_name', 'attendance_ext',
            'cbs_code', 'cost_date',
            'regular_hours', 'overtime_hours',
            'basic_rate', 'overtime_rate',
            'regular_cost', 'overtime_cost', 'total_cost',
            'posted_at',
        ]
        read_only_fields = ['regular_cost', 'overtime_cost', 'total_cost', 'posted_at']


# ---------------------------------------------------------------------------
# PRODUCTIVITY
# ---------------------------------------------------------------------------

class ProductivityRecordSerializer(serializers.ModelSerializer):
    project_code = serializers.CharField(source='project.project_code', read_only=True)
    activity_name = serializers.CharField(source='activity.activity_name', read_only=True)
    trade_name = serializers.CharField(source='trade.trade_name', read_only=True)

    class Meta:
        model = ProductivityRecord
        fields = [
            'id', 'project', 'project_code', 'activity', 'activity_name',
            'trade', 'trade_name', 'record_date',
            'man_hours_used', 'quantity_executed', 'qty_unit',
            'productivity_rate', 'entered_by', 'created_at',
        ]
        read_only_fields = ['productivity_rate', 'entered_by', 'created_at']


# ---------------------------------------------------------------------------
# CAMP MANAGEMENT
# ---------------------------------------------------------------------------

class CampAllocationSerializer(serializers.ModelSerializer):
    employee_name = serializers.CharField(source='employee.name', read_only=True)
    room_number = serializers.CharField(source='room.room_number', read_only=True)
    camp_name = serializers.CharField(source='room.camp.name', read_only=True)

    class Meta:
        model = CampAllocation
        fields = [
            'id', 'employee', 'employee_name', 'room', 'room_number',
            'camp_name', 'bed_number', 'start_date', 'end_date',
            'status', 'released_at', 'notes', 'created_at',
        ]
        read_only_fields = ['released_at', 'created_at']

    def validate(self, data):
        room = data.get('room')
        bed_number = data.get('bed_number')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # On PATCH requests not all fields are sent — skip heavy validation
        if room is None or bed_number is None or start_date is None:
            return data

        # Bed capacity check
        if room.available_beds <= 0:
            raise serializers.ValidationError(
                f'Room {room.room_number} in {room.camp.name} has no available beds.'
            )

        # Duplicate bed check
        qs = CampAllocation.objects.filter(
            room=room, bed_number=bed_number, status='Active'
        ).exclude(pk=self.instance.pk if self.instance else None)
        if end_date:
            qs = qs.filter(start_date__lte=end_date)
        qs = qs.filter(
            models.Q(end_date__isnull=True) | models.Q(end_date__gte=start_date)
        )
        if qs.exists():
            raise serializers.ValidationError(
                f'Bed {bed_number} in Room {room.room_number} is already occupied for this period.'
            )
        return data


class CampRoomSerializer(serializers.ModelSerializer):
    occupied_beds = serializers.IntegerField(read_only=True)
    available_beds = serializers.IntegerField(read_only=True)
    allocations = CampAllocationSerializer(many=True, read_only=True)

    class Meta:
        model = CampRoom
        fields = [
            'id', 'camp', 'room_number', 'bed_count',
            'occupied_beds', 'available_beds', 'allocations',
        ]


class CampSerializer(serializers.ModelSerializer):
    rooms = CampRoomSerializer(many=True, read_only=True)
    occupied_beds = serializers.IntegerField(read_only=True)
    available_beds = serializers.IntegerField(read_only=True)

    class Meta:
        model = Camp
        fields = [
            'id', 'name', 'location', 'capacity', 'is_active',
            'occupied_beds', 'available_beds', 'rooms', 'created_at',
        ]
        read_only_fields = ['created_at']


class CampListSerializer(serializers.ModelSerializer):
    """Lightweight — no nested rooms."""
    occupied_beds = serializers.IntegerField(read_only=True)
    available_beds = serializers.IntegerField(read_only=True)

    class Meta:
        model = Camp
        fields = [
            'id', 'name', 'location', 'capacity',
            'occupied_beds', 'available_beds', 'is_active',
        ]


# ---------------------------------------------------------------------------
# DASHBOARD SERIALIZERS (read-only aggregations)
# ---------------------------------------------------------------------------

class DemandVsSupplySerializer(serializers.Serializer):
    """Demand vs Supply per trade for a project."""
    trade_id = serializers.IntegerField()
    trade_name = serializers.CharField()
    required_qty = serializers.IntegerField()
    assigned_qty = serializers.IntegerField()
    gap = serializers.IntegerField()
    demand_start = serializers.DateField()
    demand_end = serializers.DateField()


class CampOccupancySerializer(serializers.Serializer):
    """Per-camp occupancy summary."""
    camp_id = serializers.IntegerField()
    camp_name = serializers.CharField()
    location = serializers.CharField()
    total_capacity = serializers.IntegerField()
    occupied = serializers.IntegerField()
    available = serializers.IntegerField()
    occupancy_percent = serializers.FloatField()


# keep this import at bottom to avoid circular references
from django.db import models
