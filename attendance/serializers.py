from rest_framework import serializers
from .models import Employee, Attendance, Site

class SiteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Site
        fields = ['id', 'name']

class UserSerializer(serializers.ModelSerializer):
    site_details = SiteSerializer(source='site', read_only=True)

    class Meta:
        model = Employee
        fields = [
            'id', 'name', 'email', 'phone', 'department', 'position', 'face_embedding', 'profile_picture',
            'job_description', 'salary_grade', 'badge_number', 'mol_id', 'labor_card_number', 'site', 'site_details', 'employer'
        ]

class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['id', 'user', 'check_in_time', 'check_out_time', 'status']


class EmployeeSerializer(serializers.ModelSerializer):
    profile_picture_url = serializers.SerializerMethodField()

    class Meta:
        model = Employee
        fields = [
            'id', 'name', 'email', 'phone', 'department', 'position', 'profile_picture_url', 'site', 
            'job_description', 'salary_grade', 'badge_number', 'mol_id', 'labor_card_number', 'employer', 
            'nationality', 'gender', 'marital_status', 'religion', 'date_of_birth', 'date_of_joining', 
            'passport_number', 'passport_expiry', 'visa_details', 'status'
        ]

    def get_profile_picture_url(self, obj):
        # This will return the absolute URL for the profile picture
        request = self.context.get('request')
        if obj.profile_picture:
            return request.build_absolute_uri(obj.profile_picture.url)
        return None

class EnrollSerializer(serializers.Serializer):
    id = serializers.IntegerField(required=False)
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField(required=False, allow_blank=True, allow_null=True)
    phone = serializers.CharField(max_length=20)
    department = serializers.CharField(max_length=100, required=False, allow_blank=True)  # Changed to CharField
    position = serializers.CharField(max_length=100, required=False, allow_blank=True)
    job_description = serializers.CharField(required=False, allow_blank=True)
    salary_grade = serializers.CharField(required=False, allow_blank=True)
    badge_number = serializers.CharField(required=False, allow_blank=True)
    mol_id = serializers.CharField(required=False, allow_blank=True)
    labor_card_number = serializers.CharField(required=False, allow_blank=True)
    site = serializers.IntegerField(required=False)
    employer = serializers.CharField(required=False, allow_blank=True)
    nationality = serializers.CharField(required=False, allow_blank=True)
    gender = serializers.CharField(required=False, allow_blank=True)
    marital_status = serializers.CharField(required=False, allow_blank=True)
    religion = serializers.CharField(required=False, allow_blank=True)
    date_of_birth = serializers.DateField(required=False, allow_null=True)
    date_of_joining = serializers.DateField(required=False, allow_null=True)
    passport_number = serializers.CharField(required=False, allow_blank=True)
    passport_expiry = serializers.DateField(required=False, allow_null=True)
    visa_details = serializers.CharField(required=False, allow_blank=True)
    status = serializers.CharField(required=False, allow_blank=True)
    # images = serializers.ListField(
    #     child=serializers.FileField(), allow_empty=False, write_only=True
    # )


class VerifySerializer(serializers.Serializer):
    slot = serializers.CharField()
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    image = serializers.ImageField()
    site_id = serializers.IntegerField(required=False, allow_null=True)