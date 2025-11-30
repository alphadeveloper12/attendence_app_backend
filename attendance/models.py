from django.db import models
from datetime import datetime, time
from django.utils import timezone

class Department(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

class Site(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

class Employee(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    position = models.CharField(max_length=50)
    face_embedding = models.JSONField(null=True)  # Store face embeddings
    profile_picture = models.ImageField(upload_to='profiles/')
    
    # New fields added
    job_description = models.TextField(null=True, blank=True)  # Job Description
    salary_grade = models.CharField(max_length=50, null=True, blank=True)  # Salary Grade/Category
    badge_number = models.CharField(max_length=20, null=True, blank=True)  # Badge Number
    mol_id = models.CharField(max_length=50, null=True, blank=True)  # MOL ID
    labor_card_number = models.CharField(max_length=50, null=True, blank=True)  # Labor Card/ Work Permit Numbers
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True)  # Site
    employer = models.CharField(max_length=100, null=True, blank=True)  # Employer (PIC or Sub contract)
    
    def __str__(self):
        return self.name
    
class Attendance(models.Model):
    user = models.ForeignKey(Employee, on_delete=models.CASCADE)
    check_in_time = models.DateTimeField(null=True, blank=True)
    break_in_time = models.DateTimeField(null=True, blank=True)
    break_out_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    late_minutes = models.IntegerField(default=0)  # Store late minutes
    early_minutes = models.IntegerField(default=0)  # Store early going minutes
    status = models.CharField(max_length=10, choices=[('present', 'Present'), ('absent', 'Absent'), ('late', 'Late')])
    date = models.DateField(auto_now_add=True)  # Track date for attendance
    latitude = models.FloatField(null=True, blank=True)  # Store latitude
    longitude = models.FloatField(null=True, blank=True)  # Store longitude
    slot = models.CharField(max_length=10, null=True, blank=True, choices=[
        ('slot1', 'Slot 1 (9-11 AM)'),
        ('slot2', 'Slot 2 (11-1 PM)'),
        ('slot3', 'Slot 3 (2-4 PM)'),
        ('slot4', 'Slot 4 (4-6 PM)'),
    ])
    
    class Meta:
        unique_together = ['user', 'date', 'slot']  # Prevent duplicate records for same user, date, and slot

    
    def __str__(self):
        return f"{self.user.name} - {self.status} ({self.date})"

    def calculate_late_and_early(self):
        """Calculate late and early minutes using timezone-aware datetimes."""

        # Expected check-in: 09:00 AM
        if self.check_in_time:
            expected_check_in = timezone.make_aware(
                datetime.combine(self.date, time(9, 0)),
                timezone.get_current_timezone()
            )
            diff = (self.check_in_time - expected_check_in).total_seconds() / 60
            self.late_minutes = max(0, int(diff))

        # Expected check-out: 06:00 PM
        if self.check_out_time:
            expected_check_out = timezone.make_aware(
                datetime.combine(self.date, time(18, 0)),
                timezone.get_current_timezone()
            )
            diff = (expected_check_out - self.check_out_time).total_seconds() / 60
            self.early_minutes = max(0, int(diff))

        self.save()

class FaceTemplate(models.Model):
    """Stores multiple embeddings for one employee."""
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='templates')
    embedding = models.JSONField()
    quality = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Template of {self.employee.name} (q={self.quality:.2f})"