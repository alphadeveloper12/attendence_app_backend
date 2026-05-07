from django.db import models
from datetime import datetime, time
from django.utils import timezone
from django.contrib.auth.models import User

class Department(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

class Site(models.Model):
    name = models.CharField(max_length=100, unique=True)
    coordinates = models.JSONField(null=True, blank=True)  # Store extracted coordinates
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
    employer = models.CharField(max_length=100, null=True, blank=True)  # Employer (PIC or Sub contract)
    
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
    status = models.CharField(max_length=50, null=True, blank=True)  # Active, Leave, Resigned, Terminated, No Renewal, Absconding, Other
    resumption_date = models.DateField(null=True, blank=True)    # Set when employee returns from Leave → Active
    last_working_date = models.DateField(null=True, blank=True)  # Set when employee leaves: Resigned/Terminated/No Renewal/Absconding
    leave_approval_date = models.DateField(null=True, blank=True)  # When the leave request was approved
    leave_start_date    = models.DateField(null=True, blank=True)  # First day of leave
    leave_end_date      = models.DateField(null=True, blank=True)  # Last day of leave (planned return on next day)

    # Additional fields
    gross_salary = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    camp = models.CharField(max_length=100, null=True, blank=True)
    transportation = models.CharField(max_length=50, null=True, blank=True, choices=[
        ('Company Bus', 'Company Bus'),
        ('personal', 'Personal'),
    ])
    basic_salary = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    category = models.CharField(max_length=20, choices=[('staff', 'Staff'), ('worker', 'Worker')], default='worker')
    
    def __str__(self):
        return self.name


class EmployeeStatusHistory(models.Model):
    """Records every status change for an employee with the dates that were captured at that moment."""
    employee            = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='status_history')
    old_status          = models.CharField(max_length=50, null=True, blank=True)
    new_status          = models.CharField(max_length=50)
    leave_approval_date = models.DateField(null=True, blank=True)
    leave_start_date    = models.DateField(null=True, blank=True)
    leave_end_date      = models.DateField(null=True, blank=True)
    resumption_date     = models.DateField(null=True, blank=True)
    last_working_date   = models.DateField(null=True, blank=True)
    note                = models.CharField(max_length=255, null=True, blank=True)
    changed_at          = models.DateTimeField(auto_now_add=True)
    changed_by          = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-changed_at']

    def __str__(self):
        return f"{self.employee.name}: {self.old_status} → {self.new_status} @ {self.changed_at:%Y-%m-%d}"


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
    status = models.CharField(max_length=10, choices=[('present', 'Present'), ('absent', 'Absent'), ('late', 'Late')])
    date = models.DateField(default=timezone.localdate)  # Track date for attendance, using localdate
    latitude = models.FloatField(null=True, blank=True)  # Store latitude
    longitude = models.FloatField(null=True, blank=True)  # Store longitude
    slot = models.CharField(max_length=20, null=True, blank=True, choices=[
        ('office_in', 'Office In'),
        ('office_out', 'Office Out'),
    ])
    is_within_geofence = models.BooleanField(default=True)  # Track if attendance was marked within geofence
    
    class Meta:
        unique_together = ['user', 'date']  # Prevent duplicate records for same user and date

    
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

        # Check if today is a special day (off day)
        day_name = self.date.strftime('%A')
        is_special_day = day_name.lower() == (day_off or "").lower()

        # Expected check-in
        if self.check_in_time:
            expected_check_in = timezone.make_aware(
                datetime.combine(self.date, start_time),
                timezone.get_current_timezone()
            )
            diff = (self.check_in_time - expected_check_in).total_seconds() / 60
            self.late_minutes = max(0, int(diff))

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
                # Normal working day
                ot_diff_seconds = (self.check_out_time - expected_check_out).total_seconds()
                if ot_diff_seconds >= 3600: # First hour threshold (3600 seconds = 1 hour)
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