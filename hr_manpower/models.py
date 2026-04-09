from django.db import models
from django.contrib.auth.models import User
from attendance.models import Employee, Site, Attendance


# ---------------------------------------------------------------------------
# 1. TRADE & SKILL LIBRARY
# ---------------------------------------------------------------------------

class Trade(models.Model):
    CATEGORY_CHOICES = [
        ('Civil', 'Civil'),
        ('MEP', 'MEP'),
        ('Factory', 'Factory'),
        ('General', 'General'),
    ]
    trade_name = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='General')
    productivity_norms = models.JSONField(
        null=True, blank=True,
        help_text='e.g. {"units_per_hour": 5, "units_per_day": 40, "unit": "m2"}'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['trade_name']

    def __str__(self):
        return f"{self.trade_name} ({self.category})"


# ---------------------------------------------------------------------------
# 2. EMPLOYEE HR PROFILE  (extends existing Employee via OneToOne — no changes to attendance app)
# ---------------------------------------------------------------------------

class EmployeeHRProfile(models.Model):
    SKILL_LEVEL_CHOICES = [
        ('Helper', 'Helper'),
        ('Skilled', 'Skilled'),
        ('Foreman', 'Foreman'),
        ('Supervisor', 'Supervisor'),
    ]
    EMPLOYMENT_TYPE_CHOICES = [
        ('Direct', 'Direct'),
        ('Subcontract', 'Subcontract'),
    ]

    employee = models.OneToOneField(
        Employee, on_delete=models.CASCADE, related_name='hr_profile'
    )
    employee_code = models.CharField(max_length=50, unique=True)
    trade = models.ForeignKey(
        Trade, on_delete=models.SET_NULL, null=True, blank=True, related_name='employees'
    )
    skill_level = models.CharField(
        max_length=20, choices=SKILL_LEVEL_CHOICES, null=True, blank=True
    )
    employment_type = models.CharField(
        max_length=20, choices=EMPLOYMENT_TYPE_CHOICES, default='Direct'
    )
    basic_rate = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True,
        help_text='Daily or hourly rate'
    )
    overtime_rate_multiplier = models.DecimalField(
        max_digits=4, decimal_places=2, default=1.50,
        help_text='OT multiplier e.g. 1.5x'
    )
    cost_center_default = models.CharField(
        max_length=100, null=True, blank=True,
        help_text='Default CBS / cost code for labor posting'
    )
    documents_dms_ids = models.JSONField(
        null=True, blank=True,
        help_text='Array of DMS document IDs: passport, visa, contract, training certs'
    )
    documents_valid = models.BooleanField(
        default=False,
        help_text='True only when all mandatory documents are uploaded and valid'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Employee HR Profile'

    def __str__(self):
        return f"{self.employee_code} – {self.employee.name}"


# ---------------------------------------------------------------------------
# 3. PROJECT & ACTIVITY
# ---------------------------------------------------------------------------

class Project(models.Model):
    STATUS_CHOICES = [
        ('Planning', 'Planning'),
        ('Active', 'Active'),
        ('On Hold', 'On Hold'),
        ('Completed', 'Completed'),
        ('Cancelled', 'Cancelled'),
    ]

    project_code = models.CharField(max_length=50, unique=True)
    project_name = models.CharField(max_length=255)
    site = models.ForeignKey(
        Site, on_delete=models.SET_NULL, null=True, blank=True, related_name='projects'
    )
    client_name = models.CharField(max_length=255, null=True, blank=True)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Planning')
    description = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='projects_created'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.project_code} – {self.project_name}"


class Activity(models.Model):
    STATUS_CHOICES = [
        ('Not Started', 'Not Started'),
        ('In Progress', 'In Progress'),
        ('Completed', 'Completed'),
        ('On Hold', 'On Hold'),
    ]

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='activities'
    )
    activity_code = models.CharField(max_length=50)
    activity_name = models.CharField(max_length=255)
    trade = models.ForeignKey(
        Trade, on_delete=models.SET_NULL, null=True, blank=True, related_name='activities'
    )
    start_date = models.DateField()
    end_date = models.DateField()
    planned_qty = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True,
        help_text='Planned quantity (m2, m3, nos, etc.)'
    )
    qty_unit = models.CharField(max_length=20, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Not Started')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['project', 'activity_code']
        ordering = ['start_date']

    def __str__(self):
        return f"{self.activity_code} – {self.activity_name}"


# ---------------------------------------------------------------------------
# 4. MANPOWER DEMAND
# ---------------------------------------------------------------------------

class ManpowerDemand(models.Model):
    SOURCE_CHOICES = [
        ('planning', 'Planning (Auto)'),
        ('manual', 'Manual'),
    ]

    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='manpower_demands'
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='manpower_demands'
    )
    trade = models.ForeignKey(
        Trade, on_delete=models.PROTECT, related_name='demands'
    )
    required_qty = models.IntegerField(help_text='Number of workers required')
    required_start_date = models.DateField()
    required_end_date = models.DateField()
    planned_man_hours = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True,
        help_text='Auto-calculated: qty × duration × hours/day'
    )
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='manual')
    notes = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['required_start_date']

    def __str__(self):
        return f"{self.project.project_code} | {self.trade.trade_name} × {self.required_qty}"

    def save(self, *args, **kwargs):
        # Auto-calculate planned man hours if not supplied
        if not self.planned_man_hours and self.required_start_date and self.required_end_date:
            days = (self.required_end_date - self.required_start_date).days + 1
            self.planned_man_hours = self.required_qty * days * 8  # assume 8 hrs/day
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# 5. PROJECT ASSIGNMENT (Allocation)
# ---------------------------------------------------------------------------

class ProjectAssignment(models.Model):
    STATUS_CHOICES = [
        ('Planned', 'Planned'),
        ('Active', 'Active'),
        ('Released', 'Released'),
    ]

    employee = models.ForeignKey(
        Employee, on_delete=models.CASCADE, related_name='project_assignments'
    )
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='assignments'
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='assignments'
    )
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    allocation_percent = models.IntegerField(
        default=100, help_text='100 = full time, 50 = half time'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Planned')
    assigned_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assignments_made'
    )
    released_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.employee.name} → {self.project.project_code} ({self.status})"


# ---------------------------------------------------------------------------
# 6. ATTENDANCE EXTENSION  (extends existing Attendance — no changes to attendance app)
# ---------------------------------------------------------------------------

class AttendanceExtension(models.Model):
    SOURCE_CHOICES = [
        ('manual', 'Manual'),
        ('mobile', 'Mobile'),
        ('biometric', 'Biometric'),
        ('csv', 'CSV Import'),
    ]
    APPROVAL_STATUS_CHOICES = [
        ('Draft', 'Draft'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]

    attendance = models.OneToOneField(
        Attendance, on_delete=models.CASCADE, related_name='hr_extension'
    )
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, null=True, blank=True, related_name='attendance_records'
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='attendance_records'
    )
    total_hours = models.DecimalField(
        max_digits=5, decimal_places=2, default=0.00,
        help_text='Total working hours for the day'
    )
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='manual')
    approval_status = models.CharField(
        max_length=20, choices=APPROVAL_STATUS_CHOICES, default='Draft'
    )
    approved_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_attendance'
    )
    approved_at = models.DateTimeField(null=True, blank=True)
    rejection_reason = models.TextField(null=True, blank=True)
    cost_posted = models.BooleanField(
        default=False, help_text='True after labor cost has been posted to ActualCost'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Attendance Extension'

    def __str__(self):
        return f"{self.attendance} – {self.approval_status}"


# ---------------------------------------------------------------------------
# 7. ACTUAL COST (Labor Cost Allocation → Finance)
# ---------------------------------------------------------------------------

class ActualCost(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='actual_costs'
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.SET_NULL, null=True, blank=True, related_name='actual_costs'
    )
    employee = models.ForeignKey(
        Employee, on_delete=models.CASCADE, related_name='actual_costs'
    )
    attendance_ext = models.OneToOneField(
        AttendanceExtension, on_delete=models.CASCADE, related_name='actual_cost'
    )
    cbs_code = models.CharField(
        max_length=100, help_text='CBS / cost center code for labor'
    )
    cost_date = models.DateField()
    regular_hours = models.DecimalField(max_digits=6, decimal_places=2, default=0.00)
    overtime_hours = models.DecimalField(max_digits=6, decimal_places=2, default=0.00)
    basic_rate = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    overtime_rate = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    regular_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    overtime_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    total_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    posted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-cost_date']

    def __str__(self):
        return f"{self.employee.name} | {self.project.project_code} | {self.cost_date} | {self.total_cost}"

    def save(self, *args, **kwargs):
        self.regular_cost = self.regular_hours * self.basic_rate
        self.overtime_cost = self.overtime_hours * self.overtime_rate
        self.total_cost = self.regular_cost + self.overtime_cost
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# 8. PRODUCTIVITY RECORD
# ---------------------------------------------------------------------------

class ProductivityRecord(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='productivity_records'
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.CASCADE, related_name='productivity_records'
    )
    trade = models.ForeignKey(
        Trade, on_delete=models.SET_NULL, null=True, blank=True, related_name='productivity_records'
    )
    record_date = models.DateField()
    man_hours_used = models.DecimalField(
        max_digits=10, decimal_places=2, default=0.00,
        help_text='Pulled from approved attendance for this activity on this date'
    )
    quantity_executed = models.DecimalField(
        max_digits=12, decimal_places=2, default=0.00,
        help_text='From QS / field entry'
    )
    qty_unit = models.CharField(max_length=20, null=True, blank=True)
    productivity_rate = models.DecimalField(
        max_digits=10, decimal_places=4, default=0.00,
        help_text='Auto-calculated: quantity_executed / man_hours_used'
    )
    entered_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['project', 'activity', 'record_date']
        ordering = ['-record_date']

    def __str__(self):
        return f"{self.activity} | {self.record_date} | rate={self.productivity_rate}"

    def save(self, *args, **kwargs):
        if self.man_hours_used and self.man_hours_used > 0:
            self.productivity_rate = round(
                float(self.quantity_executed) / float(self.man_hours_used), 4
            )
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# 9. CAMP MANAGEMENT
# ---------------------------------------------------------------------------

class Camp(models.Model):
    name = models.CharField(max_length=100, unique=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    capacity = models.IntegerField(help_text='Total bed count across all rooms')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} (cap={self.capacity})"

    @property
    def occupied_beds(self):
        return CampAllocation.objects.filter(
            room__camp=self, status='Active'
        ).count()

    @property
    def available_beds(self):
        return self.capacity - self.occupied_beds


class CampRoom(models.Model):
    camp = models.ForeignKey(Camp, on_delete=models.CASCADE, related_name='rooms')
    room_number = models.CharField(max_length=20)
    bed_count = models.IntegerField(default=4)

    class Meta:
        unique_together = ['camp', 'room_number']
        ordering = ['room_number']

    def __str__(self):
        return f"{self.camp.name} – Room {self.room_number} ({self.bed_count} beds)"

    @property
    def occupied_beds(self):
        return self.allocations.filter(status='Active').count()

    @property
    def available_beds(self):
        return self.bed_count - self.occupied_beds


class CampAllocation(models.Model):
    STATUS_CHOICES = [
        ('Active', 'Active'),
        ('Released', 'Released'),
    ]

    employee = models.ForeignKey(
        Employee, on_delete=models.CASCADE, related_name='camp_allocations'
    )
    room = models.ForeignKey(
        CampRoom, on_delete=models.CASCADE, related_name='allocations'
    )
    bed_number = models.CharField(max_length=10, help_text='e.g. "A1", "B2"')
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Active')
    released_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-start_date']

    def __str__(self):
        return (
            f"{self.employee.name} → {self.room.camp.name} "
            f"Rm{self.room.room_number}/Bed{self.bed_number} ({self.status})"
        )
