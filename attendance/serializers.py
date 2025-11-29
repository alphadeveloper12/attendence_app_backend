from rest_framework import serializers
from .models import Employee, Attendance, Department, Site

class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = ['id', 'name']

class SiteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Site
        fields = ['id', 'name']

class UserSerializer(serializers.ModelSerializer):
    department_details = DepartmentSerializer(source='department', read_only=True)
    site_details = SiteSerializer(source='site', read_only=True)

    class Meta:
        model = Employee
        fields = [
            'id', 'name', 'email', 'phone', 'department', 'department_details', 'position', 'face_embedding', 'profile_picture',
            'job_description', 'salary_grade', 'badge_number', 'mol_id', 'labor_card_number', 'site', 'site_details', 'employer'
        ]

class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['id', 'user', 'check_in_time', 'check_out_time', 'status']


class EmployeeSerializer(serializers.ModelSerializer):
    profile_picture_url = serializers.SerializerMethodField()
    department_details = DepartmentSerializer(source='department', read_only=True)

    class Meta:
        model = Employee
        fields = ['id', 'name', 'email', 'phone', 'department', 'department_details', 'position', 'profile_picture_url']

    def get_profile_picture_url(self, obj):
        # This will return the absolute URL for the profile picture
        request = self.context.get('request')
        if obj.profile_picture:
            return request.build_absolute_uri(obj.profile_picture.url)
        return None