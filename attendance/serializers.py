from rest_framework import serializers
from .models import Employee, Attendance

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = ['id', 'name', 'email', 'phone', 'department', 'position', 'face_embedding', 'profile_picture']

class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['id', 'user', 'check_in_time', 'check_out_time', 'status']


class EmployeeSerializer(serializers.ModelSerializer):
    profile_picture_url = serializers.SerializerMethodField()

    class Meta:
        model = Employee
        fields = ['id', 'name', 'email', 'phone', 'department', 'position', 'profile_picture_url']

    def get_profile_picture_url(self, obj):
        # This will return the absolute URL for the profile picture
        request = self.context.get('request')
        if obj.profile_picture:
            return request.build_absolute_uri(obj.profile_picture.url)
        return None