from django.db import models
from django.db.models import Q
from datetime import datetime, time
from django.utils import timezone
from django.contrib.auth.models import User

class Site(models.Model):
    name = models.CharField(max_length=100, unique=True)
    coordinates = models.JSONField(null=True, blank=True)  # Store extracted coordinates
    geofence_filename = models.CharField(max_length=255, null=True, blank=True)  # Original .kml filename
    geofence_lat = models.FloatField(null=True, blank=True)
    geofence_lng = models.FloatField(null=True, blank=True)
    geofence_radius_meters = models.FloatField(default=100.0)
    
    # New timing fields
    office_start_time = models.TimeField(null=True, blank=True, default="09:00:00")
    office_end_time = models.TimeField(null=True, blank=True, default="18:00:00")
    worker_start_time = models.TimeField(null=True, blank=True, default="08:00:00")
    worker_end_time = models.TimeField(null=True, blank=True, default="17:00:00")
    office_day_off = models.CharField(max_length=50, null=True, blank=True, default="Sunday")
    worker_day_off = models.CharField(max_length=50, null=True, blank=True, default="Sunday")

    def __str__(self):
        return self.name

class AdminProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='admin_profile')
    sites = models.ManyToManyField(Site, blank=True, related_name='admin_profiles')

    def __str__(self):
        site_names = ", ".join([s.name for s in self.sites.all()])
        return f"{self.user.username} - {site_names if site_names else 'No Sites'}"

class Employee(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(null=True, blank=True)
    phone = models.CharField(max_length=15, null=True, blank=True)
    department = models.CharField(max_length=100, null=True, blank=True)  # Changed to CharField
    position = models.CharField(max_length=50, null=True, blank=True)
    face_embedding = models.JSONField(null=True, blank=True)  # Store face embeddings
    profile_picture = models.ImageField(upload_to='profiles/', null=True, blank=True)
    
    # New fields added
    job_description = models.TextField(null=True, blank=True)  # Job Description
    salary_grade = models.CharField(max_length=50, null=True, blank=True)  # Salary Grade/Category
    badge_number = models.CharField(max_length=20, null=True, blank=True)  # Badge Number
    mol_id = models.CharField(max_length=50, null=True, blank=True)  # MOL ID
    labor_card_number = models.CharField(max_length=50, null=True, blank=True)  # Labor Card/ Work Permit Numbers
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True)  # Site
    # Sponsor — kept at max_length=100 because legacy free-text data (e.g. "PICDUB") lives here
    # until admins reclassify it via the dropdown. New entries use the choices below.
    sponsor = models.CharField(
        max_length=100,
        null=True, blank=True,
        choices=[
            ('Parkway', 'Parkway'),
            ('Katilink', 'Katilink'),
            ('ReadyMix', 'ReadyMix'),
            ('Mayadan', 'Mayadan'),
            ('Jafza', 'Jafza'),
            ('Golden', 'Golden'),
            ('Old Emp', 'Old Emp'),
        ],
    )
    employer = models.CharField(  # Internal employer entity
        max_length=20,
        null=True, blank=True,
        choices=[
            ('PIC', 'PIC'),
            ('KFD', 'KFD'),
            ('Kami', 'Kami'),
            ('PRMC', 'PRMC'),
        ],
    )

    # Document uploads
    passport_document = models.FileField(upload_to='employee_docs/passport/', null=True, blank=True)
    visa_document = models.FileField(upload_to='employee_docs/visa/', null=True, blank=True)
    labour_card_document = models.FileField(upload_to='employee_docs/labour/', null=True, blank=True)
    
    # New fields from Excel Import
    nationality = models.CharField(max_length=50, null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    marital_status = models.CharField(max_length=20, null=True, blank=True)
    religion = models.CharField(max_length=50, null=True, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    date_of_joining = models.DateField(null=True, blank=True)
    passport_number = models.CharField(max_length=50, null=True, blank=True)
    passport_expiry = models.DateField(null=True, blank=True)
    visa_details = models.CharField(max_length=100, null=True, blank=True)
    visa_expiry_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)  # Active, Leave, Resigned, Terminated, No Renewal, Absconding, Other
    resumption_date = models.DateField(null=True, blank=True)    # Set when employee returns from Leave → Active
    last_working_date = models.DateField(null=True, blank=True)  # Set when employee leaves: Resigned/Terminated/No Renewal/Absconding
    termination_reason = models.TextField(null=True, blank=True)  # Required when status → Resigned/Terminated
    leave_approval_date = models.DateField(null=True, blank=True)  # Repurposed: now stores Last Working Date before leave
    leave_start_date    = models.DateField(null=True, blank=True)  # First day of leave
    leave_end_date      = models.DateField(null=True, blank=True)  # Last day of leave (planned return on next day)
    leave_type          = models.CharField(max_length=30, null=True, blank=True)  # Annual / Emergency / Unpaid / Hajj-Umrah
    leave_ticket_eligible = models.BooleanField(null=True, blank=True)  # True/False/None
    leave_ticket_price    = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    # Additional fields
    gross_salary = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    camp = models.CharField(max_length=100, null=True, blank=True)
    transportation = models.CharField(max_length=50, null=True, blank=True, choices=[
        ('Company Bus', 'Company Bus'),
        ('personal', 'Personal'),
    ])
    basic_salary = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    category = models.CharField(max_length=20, choices=[('staff', 'Staff'), ('worker', 'Worker')], default='worker')

    # Salary components (Accommodation / Transport / Food / Fixed OT / Others / Salary Reduction / Remarks)
    accommodation_allowance = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    transport_allowance     = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    food_allowance          = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    fixed_ot_allowance      = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    other_allowance         = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    salary_reduction        = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    salary_remarks          = models.TextField(null=True, blank=True)

    # ── Insurance & End-of-Service ───────────────────────────────────────────
    # WC (Workmen's Compensation) insurance
    wc_insurance_name         = models.CharField(max_length=150, null=True, blank=True)
    wc_insurance_start_date   = models.DateField(null=True, blank=True)
    wc_insurance_end_date     = models.DateField(null=True, blank=True)
    wc_insurance_status       = models.CharField(max_length=20, null=True, blank=True)  # Active / Inactive / Expired
    wc_insurance_premium_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    # Medical insurance
    medical_insurance_name         = models.CharField(max_length=150, null=True, blank=True)
    medical_insurance_start_date   = models.DateField(null=True, blank=True)
    medical_insurance_end_date     = models.DateField(null=True, blank=True)
    medical_insurance_card_number  = models.CharField(max_length=80, null=True, blank=True)
    medical_insurance_status       = models.CharField(max_length=20, null=True, blank=True)
    medical_insurance_premium_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    # End of Service (EOS) — lives under the Employment Status tab; the employee's
    # status field already conveys the state, so EOS only needs subject/date/note.
    eos_subject = models.CharField(max_length=150, null=True, blank=True)
    eos_date    = models.DateField(null=True, blank=True)
    eos_note    = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name


class EmployeeStatusHistory(models.Model):
    """Records every status change for an employee with the dates that were captured at that moment."""
    employee            = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='status_history')
    old_status          = models.CharField(max_length=50, null=True, blank=True)
    new_status          = models.CharField(max_length=50)
    leave_approval_date = models.DateField(null=True, blank=True)  # Last working date before leave
    leave_start_date    = models.DateField(null=True, blank=True)
    leave_end_date      = models.DateField(null=True, blank=True)
    resumption_date     = models.DateField(null=True, blank=True)
    last_working_date   = models.DateField(null=True, blank=True)
    leave_type          = models.CharField(max_length=30, null=True, blank=True)
    leave_ticket_eligible = models.BooleanField(null=True, blank=True)
    leave_ticket_price    = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    note                = models.CharField(max_length=255, null=True, blank=True)
    changed_at          = models.DateTimeField(auto_now_add=True)
    changed_by          = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-changed_at']

    def __str__(self):
        return f"{self.employee.name}: {self.old_status} → {self.new_status} @ {self.changed_at:%Y-%m-%d}"


class EmployeeSiteHistory(models.Model):
    """Records every site assignment change for an employee.

    Each row represents a transition from `old_site` to `new_site`. The
    timeline can be derived by sorting rows by `effective_from` ascending:
    the row N row's `new_site` is the assignment until row N+1's `effective_from`.
    """
    employee       = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='site_history')
    old_site       = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    new_site       = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    effective_from = models.DateField(default=timezone.localdate)
    note           = models.CharField(max_length=255, null=True, blank=True)
    changed_at     = models.DateTimeField(auto_now_add=True)
    changed_by     = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-effective_from', '-changed_at']

    def __str__(self):
        return (
            f"{self.employee.name}: "
            f"{self.old_site.name if self.old_site else '—'} → "
            f"{self.new_site.name if self.new_site else '—'} @ {self.effective_from}"
        )


class Department(models.Model):
    """A managed department — admins create / edit / delete these from the
    Department page. All Position rows (JobCategory) link here via department_fk,
    and the modal/top-bar dropdowns + Distribution List read from this table.
    """
    name          = models.CharField(max_length=200, unique=True)
    manager_name  = models.CharField(max_length=200, null=True, blank=True)
    is_active     = models.BooleanField(default=True)
    sheet_order   = models.IntegerField(default=0)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['sheet_order', 'name']

    def __str__(self):
        return self.name


class JobCategory(models.Model):
    """A canonical job position / trade name, grouped by department + employee type.

    Originally seeded from the PIC manpower spreadsheets. Admins manage positions
    via the Department page; the modal Position dropdown + Distribution List read
    from this table.
    """
    EMP_TYPE_CHOICES = [
        ('staff', 'Staff'),
        ('worker', 'Worker'),
        ('resource', 'Manpower Resource'),
    ]
    name          = models.CharField(max_length=200)
    # Legacy text-mirror of the department label — kept so old code paths and
    # imports still work. The authoritative link is `department_fk`.
    department    = models.CharField(max_length=200, null=True, blank=True)
    department_fk = models.ForeignKey(
        Department, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='positions',
    )
    employee_type = models.CharField(max_length=10, choices=EMP_TYPE_CHOICES, default='worker')
    sheet_order   = models.IntegerField(default=0)
    is_active     = models.BooleanField(default=True)

    class Meta:
        ordering = ['employee_type', 'sheet_order', 'name']
        unique_together = [('name', 'employee_type')]

    def __str__(self):
        return f"[{self.get_employee_type_display()}] {self.department or '—'} → {self.name}"


class EmployeeAttachment(models.Model):
    """Free-form file attachments admins upload against an employee.

    Each entry has an admin-supplied name (e.g. "Driving Licence", "Contract",
    "Emirates ID") plus the file itself. There is no schema beyond that — admins
    can attach as many files as needed.
    """
    employee   = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='attachments')
    name       = models.CharField(max_length=200)
    file       = models.FileField(upload_to='employee_attachments/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.employee.name} — {self.name}"


class EmployeeSalaryHistory(models.Model):
    """Snapshot of an employee's salary components after each increment / change.

    A row stores the *new* component values plus an optional remarks note and
    the effective date. The previous values can be read by looking at the row
    immediately before this one (or from the Employee row itself for the initial
    state, although we usually log an explicit initial row too).
    """
    employee     = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='salary_history')
    basic_salary            = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    accommodation_allowance = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    transport_allowance     = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    food_allowance          = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    fixed_ot_allowance      = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    other_allowance         = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    salary_reduction        = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    gross_salary            = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    remarks                 = models.TextField(null=True, blank=True)
    effective_from          = models.DateField(default=timezone.localdate)
    changed_at              = models.DateTimeField(auto_now_add=True)
    changed_by              = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-effective_from', '-changed_at']

    def __str__(self):
        return f"{self.employee.name} salary @ {self.effective_from}"


class Attendance(models.Model):
    user = models.ForeignKey(Employee, on_delete=models.CASCADE)
    check_in_time = models.DateTimeField(null=True, blank=True)
    break_in_time = models.DateTimeField(null=True, blank=True)
    break_out_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    late_minutes = models.IntegerField(default=0)  # Store late minutes
    early_minutes = models.IntegerField(default=0)  # Store early going minutes
    normal_ot_hours = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)
    special_ot_hours = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)
    status = models.CharField(max_length=10, choices=[('present', 'Present'), ('absent', 'Absent'), ('late', 'Late'), ('sick', 'Sick'), ('leave', 'Leave')])
    date = models.DateField(default=timezone.localdate)  # Track date for attendance, using localdate
    latitude = models.FloatField(null=True, blank=True)  # Store latitude
    longitude = models.FloatField(null=True, blank=True)  # Store longitude
    slot = models.CharField(max_length=20, null=True, blank=True, choices=[
        ('office_in', 'Office In'),
        ('office_out', 'Office Out'),
    ])
    is_within_geofence = models.BooleanField(default=True)  # Track if attendance was marked within geofence
    medical_certificate = models.FileField(upload_to='medical_certificates/', null=True, blank=True)
    sick_leave_note = models.TextField(null=True, blank=True)
    sick_leave_marked_at = models.DateTimeField(null=True, blank=True)
    sick_leave_marked_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='sick_leaves_marked')
    
    class Meta:
        unique_together = ['user', 'date']  # Prevent duplicate records for same user and date
        indexes = [
            # Speeds up "checked out without a check-in" lookups (Missing Check-In),
            # which otherwise scan the whole attendance table.
            models.Index(
                fields=['-date'],
                name='att_missing_checkin_idx',
                condition=Q(check_in_time__isnull=True, check_out_time__isnull=False),
            ),
            models.Index(fields=['date'], name='att_date_idx'),
        ]


    def __str__(self):
        return f"{self.user.name} - {self.status} ({self.date})"

    def calculate_late_and_early(self):
        """Calculate late, early minutes and overtime hours including special OT."""
        from datetime import datetime
        if not self.user or not self.user.site:
            return

        site = self.user.site
        is_staff = self.user.category == 'staff'
        
        # Determine schedule based on category
        start_time = site.office_start_time if is_staff else site.worker_start_time
        end_time = site.office_end_time if is_staff else site.worker_end_time
        day_off = site.office_day_off if is_staff else site.worker_day_off
        
        if not start_time or not end_time:
            return

        # Org-wide config (late grace, OT threshold, weekend days, holidays).
        # Defaults keep legacy behaviour if the settings row doesn't exist yet.
        try:
            cfg = AppSettings.load()
            grace_minutes = cfg.late_grace_minutes or 0
            ot_threshold_seconds = (cfg.normal_ot_threshold_minutes or 60) * 60
            weekend_days = [d.lower() for d in (cfg.weekend_days or [])]
        except Exception:
            cfg = None
            grace_minutes = 0
            ot_threshold_seconds = 3600
            weekend_days = []

        # A "special day" (all hours = special OT) is the site day-off, a
        # configured weekend day, or an org public holiday.
        day_name = self.date.strftime('%A')
        is_special_day = day_name.lower() == (day_off or "").lower()
        if not is_special_day and day_name.lower() in weekend_days:
            is_special_day = True
        if not is_special_day:
            try:
                if PublicHoliday.objects.filter(date=self.date).exists() or \
                   PublicHoliday.objects.filter(
                       recurring_annually=True,
                       date__month=self.date.month, date__day=self.date.day).exists():
                    is_special_day = True
            except Exception:
                pass

        # Expected check-in
        if self.check_in_time:
            expected_check_in = timezone.make_aware(
                datetime.combine(self.date, start_time),
                timezone.get_current_timezone()
            )
            diff = (self.check_in_time - expected_check_in).total_seconds() / 60
            # Within the grace period → not late.
            self.late_minutes = max(0, int(diff - grace_minutes))

        # Expected check-out and Overtime
        if self.check_out_time:
            expected_check_out = timezone.make_aware(
                datetime.combine(self.date, end_time),
                timezone.get_current_timezone()
            )
            
            # Early minutes
            early_diff = (expected_check_out - self.check_out_time).total_seconds() / 60
            self.early_minutes = max(0, int(early_diff))
            
            # Overtime Calculation
            # 1. Normal OT: check_out > expected_check_out + 1 hour (on working days)
            # 2. Special OT: any work on is_special_day
            
            if is_special_day:
                # All work on special day is Special OT
                if self.check_in_time and self.check_out_time:
                    total_work_minutes = (self.check_out_time - self.check_in_time).total_seconds() / 3600
                    self.special_ot_hours = round(max(0.0, total_work_minutes), 2)
                    self.normal_ot_hours = 0.0
            else:
                # Normal working day — OT only counts past the configured threshold.
                ot_diff_seconds = (self.check_out_time - expected_check_out).total_seconds()
                if ot_diff_seconds >= ot_threshold_seconds:
                    self.normal_ot_hours = round(ot_diff_seconds / 3600, 2)
                else:
                    self.normal_ot_hours = 0.0
                self.special_ot_hours = 0.0

        # self.save()  <-- Removed to prevent double-save in view

class FaceTemplate(models.Model):
    """Stores multiple embeddings for one employee."""
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='templates')
    embedding = models.JSONField()
    quality = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Template of {self.employee.name} (q={self.quality:.2f})"

class AppBuild(models.Model):
    APP_CHOICES = [
        ('attendance', 'Attendance App'),
        ('fuel', 'Fuel App'),
    ]
    app_type = models.CharField(max_length=20, choices=APP_CHOICES, unique=True)
    file = models.FileField(upload_to='builds/')
    uploaded_at = models.DateTimeField(auto_now=True)
    version = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"{self.get_app_type_display()} - {self.uploaded_at}"


# ── Global configuration ──────────────────────────────────────────────────
# Default factories for JSON fields. Kept as named module-level functions so
# migrations stay stable (lambdas can't be serialized by Django migrations).
def _default_gross_formula():
    """Sign of each component in the Gross Salary sum. +1 adds, -1 subtracts."""
    return {
        'basic_salary': 1,
        'accommodation_allowance': 1,
        'transport_allowance': 1,
        'food_allowance': 1,
        'fixed_ot_allowance': 1,
        'other_allowance': 1,
        'salary_reduction': -1,
    }


def _default_sponsors():
    return ['Parkway', 'Katilink', 'ReadyMix', 'Mayadan', 'Jafza', 'Golden', 'Old Emp']


def _default_employers():
    return ['PIC', 'KFD', 'Kami', 'PRMC']


def _default_weekend_days():
    return ['Friday', 'Saturday']


def _default_nav_visibility():
    """Nav links visible to NON-superuser admins. Superusers always see all.
    Missing keys default to visible (True)."""
    return {
        'dashboard': True, 'reports': True, 'monthly_report': True,
        'distribution_list': True, 'departments': True, 'user_face': True,
        'attrition_risk': True, 'document_expiry': True, 'manpower_recs': True,
        'ask_data': True, 'geofence_tuning': True, 'salary': True,
        'sites': True, 'site_admins': True, 'settings': False,
    }


class AppSettings(models.Model):
    """Singleton holding org-wide configuration (always pk=1). Use
    AppSettings.load() to fetch/create it."""

    # --- 1. Attendance & shift rules (org defaults; per-site values live on Site) ---
    default_office_start_time   = models.TimeField(default="09:00:00")
    default_office_end_time     = models.TimeField(default="18:00:00")
    default_worker_start_time   = models.TimeField(default="08:00:00")
    default_worker_end_time     = models.TimeField(default="17:00:00")
    default_office_day_off      = models.CharField(max_length=20, default="Sunday")
    default_worker_day_off      = models.CharField(max_length=20, default="Sunday")
    late_grace_minutes          = models.PositiveIntegerField(default=0)
    half_day_threshold_hours    = models.DecimalField(max_digits=4, decimal_places=2, default=4.00)
    normal_ot_threshold_minutes = models.PositiveIntegerField(default=60)
    weekend_days                = models.JSONField(default=_default_weekend_days, blank=True)

    # --- 2. Geofence / location ---
    default_geofence_radius_meters = models.FloatField(default=100.0)
    gps_accuracy_tolerance_meters  = models.FloatField(default=50.0)

    # --- 4. Master lists (only the ones without their own model) ---
    sponsors  = models.JSONField(default=_default_sponsors, blank=True)
    employers = models.JSONField(default=_default_employers, blank=True)

    # --- 5. Salary ---
    currency_code         = models.CharField(max_length=8, default="AED")
    salary_superuser_only = models.BooleanField(default=True)
    gross_formula         = models.JSONField(default=_default_gross_formula, blank=True)

    # --- 6. Documents & compliance ---
    passport_reminder_lead_days    = models.PositiveIntegerField(default=60)
    visa_reminder_lead_days        = models.PositiveIntegerField(default=60)
    labour_card_reminder_lead_days = models.PositiveIntegerField(default=60)
    mol_reminder_lead_days         = models.PositiveIntegerField(default=60)
    expiry_alert_recipients        = models.JSONField(default=list, blank=True)

    # --- Navigation visibility (for non-superuser admins) ---
    nav_visibility = models.JSONField(default=_default_nav_visibility, blank=True)

    # --- Data retention ---
    # How many months of daily Distribution List snapshots to keep. 0 = keep forever.
    distribution_snapshot_retention_months = models.PositiveIntegerField(default=12)

    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        verbose_name = "App Settings"
        verbose_name_plural = "App Settings"

    def __str__(self):
        return "App Settings"

    def save(self, *args, **kwargs):
        self.pk = 1  # enforce singleton
        super().save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class PublicHoliday(models.Model):
    """Org-wide public holidays — treated as off/special-OT days by the engine."""
    name               = models.CharField(max_length=120)
    date               = models.DateField()
    recurring_annually = models.BooleanField(default=False)
    created_at         = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['date']

    def __str__(self):
        return f"{self.name} ({self.date})"


class DistributionSnapshot(models.Model):
    """A frozen copy of the Distribution List for one day and one tab type.

    The live distribution is computed from the current Employee table, which has
    no per-field history (department/position aren't versioned). To let admins
    view past-date distributions we store the fully-computed payload once per day
    per type ('resource' / 'staff' / 'worker'). Reading a past date returns the
    stored snapshot, so what you see is exactly what the roster looked like then.
    """
    date      = models.DateField()
    dist_type = models.CharField(max_length=20)   # 'resource' | 'staff' | 'worker'
    payload   = models.JSONField()                # same shape the live API returns
    created_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('date', 'dist_type')
        ordering = ['-date']
        indexes = [models.Index(fields=['dist_type', 'date'])]

    def __str__(self):
        return f"Distribution {self.dist_type} @ {self.date}"