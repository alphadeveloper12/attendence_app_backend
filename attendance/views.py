import logging
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from rest_framework.permissions import AllowAny
from .serializers import EmployeeSerializer, DepartmentSerializer, SiteSerializer
from django.conf import settings
from .models import Employee, Attendance, Department, Site
from .serializers import UserSerializer, AttendanceSerializer

class DepartmentListView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        departments = Department.objects.all()
        serializer = DepartmentSerializer(departments, many=True)
        return Response(serializer.data)

class SiteListView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        sites = Site.objects.all()
        serializer = SiteSerializer(sites, many=True)
        return Response(serializer.data)
from deepface import DeepFace
from django.core.files.base import ContentFile
import re
from datetime import datetime
from rest_framework.permissions import IsAdminUser
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from rest_framework.permissions import IsAdminUser
import gc
from tensorflow.keras import backend as K
from scipy.spatial.distance import cosine


# Set up logging
logger = logging.getLogger(__name__)

@permission_classes([IsAdminUser])
class RegisterUserView(APIView):
    def post(self, request):
        print(request.data)
        try:
            # Retrieve incoming data
            data = request.data
            face_image = data.get('faceImage')  # Base64 encoded image
            user_data = data.get('user_data')  # User data object from frontend
            
            # Check if user_data exists
            if not user_data:
                return Response(
                    {"error": "Missing user_data object"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not face_image:
                 return Response(
                    {"error": "Missing faceImage"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract user data from the nested user_data object
            name = user_data.get('name')
            email = user_data.get('email')
            phone = user_data.get('phone')
            department_id = user_data.get('department')
            position = user_data.get('position')
            job_description = user_data.get('job_description')
            salary_grade = user_data.get('salary_grade')
            badge_number = user_data.get('badge_number')
            mol_id = user_data.get('mol_id')
            labor_card_number = user_data.get('labor_card_number')
            site_id = user_data.get('site')
            employer = user_data.get('employer')

            logger.info(f"Registering user: {name} with email: {email}")

            # Fetch Department and Site objects
            department_obj = None
            if department_id:
                try:
                    department_obj = Department.objects.get(id=department_id)
                except Department.DoesNotExist:
                    return Response({"error": "Invalid department ID"}, status=status.HTTP_400_BAD_REQUEST)

            site_obj = None
            if site_id:
                try:
                    site_obj = Site.objects.get(id=site_id)
                except Site.DoesNotExist:
                    return Response({"error": "Invalid site ID"}, status=status.HTTP_400_BAD_REQUEST)

            # Check if email already exists
            if Employee.objects.filter(email=email).exists():
                return Response(
                    {"error": "User with this email already exists"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 1. Clean base64 string
            base64_pattern = r"^data:image\/(jpeg|png);base64,"
            face_image_data = re.sub(base64_pattern, "", face_image)

            if not face_image_data:
                return Response({"error": "Invalid base64 image format"}, status=status.HTTP_400_BAD_REQUEST)

            # 2. Decode and save temp file
            decoded_image = base64.b64decode(face_image_data)
            img_file = ContentFile(decoded_image, name="user_face.jpg")

            # Create user object temporarily to save image
            new_user = Employee(
                name=name,
                email=email,
                phone=phone,
                department=department_obj,
                position=position,
                job_description=job_description,
                salary_grade=salary_grade,
                badge_number=badge_number,
                mol_id=mol_id,
                labor_card_number=labor_card_number,
                site=site_obj,
                employer=employer,
            )
            
            # Save image to get path
            new_user.profile_picture.save("temp_register_face.jpg", img_file)
            img_path = new_user.profile_picture.path

            # 3. Generate embedding using DeepFace (VGG-Face)
            try:
                embeddings = DeepFace.represent(
                    img_path, 
                    model_name="VGG-Face",
                    enforce_detection=False
                )
            except Exception as e:
                new_user.delete() # Cleanup
                logger.error(f"DeepFace error: {str(e)}")
                return Response({"error": f"Error processing face: {str(e)}"}, status=400)

            if not embeddings:
                new_user.delete()
                return Response({"error": "No face detected in image"}, status=400)

            face_embedding = embeddings[0].get("embedding")
            
            if not face_embedding:
                new_user.delete()
                return Response({"error": "Face embedding could not be generated."}, status=400)

            # 4. Save Full User with embedding
            new_user.face_embedding = face_embedding
            new_user.save()

            logger.info(f"USER_REGISTERED_SUCCESSFULLY: {name}")

            # Build response with the necessary fields
            response_data = {
                "message": "User registered successfully!",
                "user": UserSerializer(new_user).data,
            }

            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"ERROR => {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class MarkAttendanceView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            print(request.data)
            # Retrieve incoming data
            face_image = request.data.get('faceImage')
            slot = request.data.get('slot')
            latitude = request.data.get('latitude')
            longitude = request.data.get('longitude')

            # Log incoming request data
            logger.info(f"Request Data: {request.data}")

            # Validate that all required fields are present
            missing_fields = []
            if not face_image:
                missing_fields.append('face_image')
            if not slot:
                missing_fields.append('slot')
            if latitude is None:
                missing_fields.append('latitude')
            if longitude is None:
                missing_fields.append('longitude')

            if missing_fields:
                return Response(
                    {"error": f"Missing fields: {', '.join(missing_fields)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Clean base64 string
            base64_pattern = r"^data:image\/(jpeg|png);base64,"
            face_image_data = re.sub(base64_pattern, "", face_image)

            if not face_image_data:
                return Response({"error": "Invalid base64 image format"}, status=status.HTTP_400_BAD_REQUEST)

            # Decode the image
            decoded_image = base64.b64decode(face_image_data)
            img_file = ContentFile(decoded_image, name="user_face.jpg")

            # Save image temporarily to get a file path for DeepFace
            temp_user = Employee.objects.first()
            if not temp_user:
                return Response({"error": "No employee found in the system"}, status=status.HTTP_400_BAD_REQUEST)

            temp_user.profile_picture.save("temp_face.jpg", img_file)
            img_path = temp_user.profile_picture.path

            # Generate face embeddings using DeepFace
            try:
                embeddings = DeepFace.represent(
                    img_path, 
                    model_name="VGG-Face",
                    enforce_detection=False
                )
            except Exception as e:
                logger.error(f"DeepFace error: {str(e)}")
                return Response({"error": f"Error processing face: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

            # Cleanup temp file
            if temp_user.profile_picture:
                temp_user.profile_picture.delete()

            if not embeddings:
                return Response({"error": "No face detected in image"}, status=400)

            new_embedding = embeddings[0].get("embedding")

            if not new_embedding:
                return Response({"error": "Face embedding not found"}, status=400)

            # Compare with stored users
            matched_user = None
            for user in Employee.objects.all():
                if user.face_embedding and self.compare_embeddings(user.face_embedding, new_embedding):
                    matched_user = user
                    break

            if not matched_user:
                return Response({"error": "User not found or face not recognized"}, status=400)

            # Mark Attendance
            now = timezone.localtime()
            today = now.date()
            
            # Check if attendance already exists for today
            attendance = Attendance.objects.filter(user=matched_user, date=today).first()
            
            if not attendance:
                attendance = Attendance.objects.create(
                    user=matched_user,
                    date=today,
                    status='present',
                    latitude=latitude,
                    longitude=longitude
                )
            else:
                attendance.status = 'present'
                attendance.latitude = latitude
                attendance.longitude = longitude
                attendance.save()

            # Update slot time
            if slot == 'office_in':
                attendance.check_in_time = now
            elif slot == 'break_in':
                attendance.break_in_time = now
            elif slot == 'break_out':
                attendance.break_out_time = now
            elif slot == 'office_out':
                attendance.check_out_time = now

            attendance.calculate_late_and_early()

            # Prepare response
            response_time = None
            if slot == 'office_in' and attendance.check_in_time:
                response_time = attendance.check_in_time.strftime('%I:%M %p')
            elif slot == 'break_in' and attendance.break_in_time:
                response_time = attendance.break_in_time.strftime('%I:%M %p')
            elif slot == 'break_out' and attendance.break_out_time:
                response_time = attendance.break_out_time.strftime('%I:%M %p')
            elif slot == 'office_out' and attendance.check_out_time:
                response_time = attendance.check_out_time.strftime('%I:%M %p')

            return Response({
                "message": f"Attendance marked successfully for {slot}.",
                "user": matched_user.name,
                "slot": slot,
                "time": response_time,
                "late_minutes": attendance.late_minutes,
                "early_minutes": attendance.early_minutes,
                "latitude": attendance.latitude,
                "longitude": attendance.longitude
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def compare_embeddings(self, stored_embedding, new_embedding, threshold=0.5):
        """
        Compare two face embeddings and return True if they are similar within a threshold.
        """
        # Check if dimensions match
        if len(stored_embedding) != len(new_embedding):
            return False

        distance = cosine(stored_embedding, new_embedding)
        return distance < threshold

class AdminLoginView(APIView):
    """
    Admin Login using email and password to get JWT token.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        print(email, password)

        if not email or not password:
            return Response(
                {"error": "Email and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"error": "Admin with this email does not exist."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not user.is_superuser:
            return Response(
                {"error": "You must be an admin to login."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user = authenticate(request, username=user.username, password=password)

        if user is not None:
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            return Response(
                {"message": "Login successful", "access_token": str(access_token)},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"error": "Invalid email or password"},
                status=status.HTTP_400_BAD_REQUEST,
            )



class AttendanceStatsView(APIView):
    """
    View to get the total number of employees and today's attendance count.
    """
    permission_classes = [AllowAny]
    def get(self, request):
        try:
            total_employees = Employee.objects.count()
            today = timezone.localdate()
            today_attendance_count = Attendance.objects.filter(date=today).count()

            return Response({
                "total_employees": total_employees,
                "today_attendance_count": today_attendance_count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

@permission_classes([IsAdminUser])
class EmployeeListView(APIView):
    def get(self, request):
        employees = Employee.objects.all()
        serializer = EmployeeSerializer(employees, many=True, context={'request': request})
        return Response(serializer.data)


# Admin Dashboard Views (Django Template Views)
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from datetime import timedelta
from collections import defaultdict

def admin_login_view(request):
    """Admin login view"""
    if request.user.is_authenticated:
        return redirect('admin-dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            return redirect('admin-dashboard')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials or not an admin user'})
    
    return render(request, 'login.html')

@login_required(login_url='admin-login')
def admin_dashboard_view(request):
    """Main dashboard showing users grouped by site"""
    if not request.user.is_staff:
        return redirect('admin-login')
    
    # Get all employees
    employees = Employee.objects.select_related('site', 'department').all()
    
    # Group employees by site
    employees_by_site = defaultdict(list)
    for employee in employees:
        site_name = employee.site.name if employee.site else None
        employees_by_site[site_name].append(employee)
    
    # Get statistics
    total_employees = employees.count()
    total_sites = Site.objects.count()
    total_departments = Department.objects.count()
    today = timezone.now().date()
    today_attendance = Attendance.objects.filter(date=today).count()
    
    context = {
        'employees_by_site': dict(employees_by_site),
        'total_employees': total_employees,
        'total_sites': total_sites,
        'total_departments': total_departments,
        'today_attendance': today_attendance,
    }
    
    return render(request, 'dashboard.html', context)

@login_required(login_url='admin-login')
def admin_user_detail_view(request, user_id):
    """User detail view with attendance records"""
    if not request.user.is_staff:
        return redirect('admin-login')
    
    employee = get_object_or_404(Employee, id=user_id)
    
    # Get filter parameter (default to daily)
    filter_type = request.GET.get('filter', 'daily')
    today = timezone.now().date()
    
    # Define 4 time slots
    SLOTS = {
        'Slot 1': {'time_range': '9:00 AM - 11:00 AM', 'slot_value': 'slot1'},
        'Slot 2': {'time_range': '11:00 AM - 1:00 PM', 'slot_value': 'slot2'},
        'Slot 3': {'time_range': '2:00 PM - 4:00 PM', 'slot_value': 'slot3'},
        'Slot 4': {'time_range': '4:00 PM - 6:00 PM', 'slot_value': 'slot4'},
    }
    
    context = {
        'employee': employee,
        'filter': filter_type,
        'today': today,
    }
    
    if filter_type == 'daily':
        # Daily view: Show 4 slots for today
        attendance_records = Attendance.objects.filter(
            user=employee,
            date=today
        )
        
        # Organize records by slot
        slots_data = {}
        for slot_name, slot_info in SLOTS.items():
            slot_record = attendance_records.filter(slot=slot_info['slot_value']).first()
            slots_data[slot_name] = {
                'time_range': slot_info['time_range'],
                'status': slot_record.status if slot_record else None,
                'check_in': slot_record.check_in_time if slot_record else None,
                'late_minutes': slot_record.late_minutes if slot_record else 0,
                'latitude': slot_record.latitude if slot_record else None,
                'longitude': slot_record.longitude if slot_record else None,
            }
        
        context['slots'] = slots_data
        context['total_records'] = attendance_records.count()
        context['present_count'] = attendance_records.filter(status='present').count()
        context['late_count'] = attendance_records.filter(status='late').count()
        context['absent_count'] = attendance_records.filter(status='absent').count()
        
    elif filter_type == 'weekly':
        # Weekly view: Calendar grid
        start_of_week = today - timedelta(days=today.weekday())  # Monday
        end_of_week = start_of_week + timedelta(days=6)  # Sunday
        
        attendance_records = Attendance.objects.filter(
            user=employee,
            date__range=[start_of_week, end_of_week]
        )
        
        # Create calendar structure
        calendar_days = []
        current_date = start_of_week
        
        while current_date <= end_of_week:
            day_records = attendance_records.filter(date=current_date)
            
            # Get slots for this day
            day_slots = []
            for slot_name, slot_info in SLOTS.items():
                slot_record = day_records.filter(slot=slot_info['slot_value']).first()
                day_slots.append({
                    'name': slot_name.replace('Slot ', 'S'),  # Abbreviated for display
                    'status': slot_record.status if slot_record else 'empty'
                })
            
            calendar_days.append({
                'date': current_date,
                'slots': day_slots
            })
            
            current_date += timedelta(days=1)
        
        context['calendar_days'] = calendar_days
        context['week_start'] = start_of_week
        context['week_end'] = end_of_week
        context['total_records'] = attendance_records.count()
        context['present_count'] = attendance_records.filter(status='present').count()
        context['late_count'] = attendance_records.filter(status='late').count()
        context['absent_count'] = attendance_records.filter(status='absent').count()
        
    else:
        # Monthly view: Table format
        attendance_records = Attendance.objects.filter(
            user=employee,
            date__month=today.month,
            date__year=today.year
        ).order_by('-date', 'slot')
        
        context['attendance_records'] = attendance_records
        context['total_records'] = attendance_records.count()
        context['present_count'] = attendance_records.filter(status='present').count()
        context['late_count'] = attendance_records.filter(status='late').count()
        context['absent_count'] = attendance_records.filter(status='absent').count()
    
    return render(request, 'user_detail.html', context)


@login_required(login_url='admin-login')
def admin_logout_view(request):
    """Logout view"""
    logout(request)
    return redirect('admin-login')