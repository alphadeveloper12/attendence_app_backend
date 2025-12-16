import os
import django
from django.test import RequestFactory
from django.utils import timezone
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from django.contrib.auth.models import User
from attendance.models import Site, Employee, Attendance
from attendance.views import admin_reports_view

def test_reports_map():
    # Setup Data
    user, _ = User.objects.get_or_create(username='map_test_admin', email='map@admin.com')
    user.set_password('password')
    user.is_superuser = True
    user.is_staff = True
    user.save()

    site, _ = Site.objects.get_or_create(name='Map Test Site')
    # Mock coordinates (not used in view logic but good for completeness)
    site.coordinates = [{'lat': 25.0, 'lng': 55.0}, {'lat': 25.1, 'lng': 55.1}] 
    site.save()

    employee, _ = Employee.objects.get_or_create(
        name='Map Test Employee',
        site=site,
        badge_number='MAP001'
    )

    # Create Attendance with Location
    Attendance.objects.filter(user=employee, date=timezone.localdate()).delete() # Cleanup
    Attendance.objects.create(
        user=employee,
        status='present',
        check_in_time=timezone.now(),
        latitude=25.05,
        longitude=55.05
    )

    # Request
    factory = RequestFactory()
    request = factory.get('/dashboard/reports/')
    request.user = user
    response = admin_reports_view(request)

    print(f"Response Status: {response.status_code}")

    if response.status_code == 200:
        content = response.content.decode('utf-8')
        
        # Check for View Button
        expected_onclick = f"viewLocation({site.id}, 25.05, 55.05)"
        if expected_onclick in content:
            print("SUCCESS: View Location button found with correct coordinates.")
        else:
            print(f"FAILURE: View Location button NOT found. Expected: {expected_onclick}")
            # Debug: print relevant part of content
            if 'Map Test Employee' in content:
                print("Employee found in report.")
            else:
                print("Employee NOT found in report.")
    else:
        print("FAILURE: View returned non-200 status.")

if __name__ == '__main__':
    test_reports_map()
