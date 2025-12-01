# attendance/views.py
import logging
import base64
import re
from datetime import datetime, timedelta
from collections import defaultdict

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.decorators import permission_classes

from django.contrib.auth.models import User
import numpy as np
from .models import Employee, Attendance, Department, Site, FaceTemplate

# --- NEW: our engine/utils ---
from .engine import ENGINE
from .utils import (
    dataurl_to_bytes, pil_to_bgr_array_from_bytes,
    THRESH, MARGIN, get_image_bytes
)
from .serializers import *
from rest_framework_simplejwt.tokens import RefreshToken

logger = logging.getLogger(__name__)

# ------------------ Departments / Sites (unchanged) ------------------

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

# ------------------ Register User (DeepFace -> InsightFace) ------------------

class RegisterUserView(APIView):
    permission_classes = [AllowAny]

    @transaction.atomic
    def post(self, request):
        # Validate text fields only
        s = EnrollSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        # Files MUST come from request.FILES
        files = request.FILES.getlist('images') or request.FILES.getlist('images[]')

        if not files:
            return Response(
                {"error": "No images uploaded. Use form-data with key 'images' and attach files."},
                status=400
            )

        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        department_id = data.get("department")
        position = data.get("position") or ""
        job_description = data.get("job_description") or ""
        salary_grade = data.get("salary_grade") or ""
        badge_number = data.get("badge_number") or ""
        mol_id = data.get("mol_id") or ""
        labor_card_number = data.get("labor_card_number") or ""
        site_id = data.get("site")
        employer = data.get("employer") or ""

        if Employee.objects.filter(email=email).exists():
            return Response({"error": "Employee with this email already exists."}, status=400)

        department_obj = Department.objects.filter(id=department_id).first() if department_id else None
        site_obj = Site.objects.filter(id=site_id).first() if site_id else None

        emp = Employee.objects.create(
            name=name, email=email, phone=phone,
            department=department_obj, position=position,
            job_description=job_description, salary_grade=salary_grade,
            badge_number=badge_number, mol_id=mol_id,
            labor_card_number=labor_card_number, site=site_obj, employer=employer
        )

        valid_vecs, rejected = [], []

        for f in files:
            try:
                f.seek(0)
                raw = f.read()
                if not raw:
                    rejected.append({"ok": False, "reason": "empty_file"})
                    continue

                # Convert and embed
                bgr = pil_to_bgr_array_from_bytes(raw)
                v, q, meta = ENGINE.embed_best_face(bgr)

                if v is None or not meta.get("ok", False):
                    rejected.append(meta)
                    continue

                FaceTemplate.objects.create(employee=emp, embedding=v.tolist(), quality=float(q))
                valid_vecs.append(v)
            except Exception as e:
                rejected.append({"ok": False, "reason": str(e)})

        if not valid_vecs:
            emp.delete()
            return Response({
                "status": "error",
                "message": "No valid faces detected in any uploaded images.",
                "rejected": rejected,
            }, status=422)

        # Compute centroid and store profile pic
        centroid = np.mean(valid_vecs, axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        emp.face_embedding = centroid.tolist()
        first_file = files[0]
        first_file.seek(0)
        emp.profile_picture.save(f"{emp.id}_profile_{first_file.name}", first_file, save=False)
        emp.save(update_fields=["face_embedding", "profile_picture"])

        # Rebuild FAISS index
        qs = FaceTemplate.objects.all().only("id", "employee_id", "embedding")
        tuples = [(t.id, t.employee_id, np.array(t.embedding, dtype=np.float32)) for t in qs]
        ENGINE.rebuild_index(tuples)

        return Response({
            "status": "success",
            "message": f"Employee '{emp.name}' enrolled successfully.",
            "templates_added": len(valid_vecs),
            "rejected": rejected,
            "employee": UserSerializer(emp, context={'request': request}).data
        }, status=201)

# ------------------ Mark Attendance (DeepFace -> InsightFace + FAISS) ------------------

class MarkAttendanceView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # Validate non-file fields first (slot/lat/long)
        s = VerifySerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        slot = data["slot"]
        latitude = data["latitude"]
        longitude = data["longitude"]

        # File MUST come from request.FILES (avoid serializer coercion)
        img = request.FILES.get('image') or request.FILES.get('image[]')
        if not img:
            return Response({"error": "No image uploaded. Use form-data with key 'image'."}, status=400)

        # Read bytes exactly once
        img.seek(0)
        raw = img.read()
        if not raw:
            return Response({"error": "Uploaded image is empty."}, status=400)

        # Convert to BGR ndarray and embed
        try:
            bgr = pil_to_bgr_array_from_bytes(raw)
        except Exception as e:
            return Response({"error": f"Invalid image file. {e}"}, status=400)

        v, q, meta = ENGINE.embed_best_face(bgr)
        if v is None:
            return Response({"error": "No face detected in image."}, status=400)

        # Quality gates (soft blur passes; others return actionable messages)
        if not meta.get("ok", False) and not str(meta.get("reason", "")).startswith("soft_blurry"):
            reason = str(meta.get("reason", ""))
            if "too_dark" in reason:
                return Response({"error": "Lighting too dim. Please brighten the environment."}, status=422)
            if "too_bright" in reason:
                return Response({"error": "Image too bright. Avoid direct glare."}, status=422)
            if "face_too_small" in reason:
                return Response({"error": "Move closer to the camera."}, status=422)
            if "det_score" in reason or "blurry" in reason:
                return Response({"error": "Face not clear. Hold still and retry."}, status=422)

        # Gallery must exist
        if ENGINE.index is None or len(ENGINE.ids) == 0:
            return Response({"error": "No enrolled employees in gallery."}, status=400)

        # FAISS nearest neighbors (cosine similarity on L2-normalized vectors)
        sims, idxs = ENGINE.search(v, k=10)
        rows = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            template_id, employee_id = ENGINE.ids[idx]
            rows.append((template_id, employee_id, float(sim)))

        if not rows:
            return Response({"error": "No match found."}, status=400)

        # Aggregate to best per employee
        per_emp = {}
        for _, eid, sim in rows:
            if eid not in per_emp or sim > per_emp[eid]:
                per_emp[eid] = sim

        # Decide winner with threshold + margin
        ranked = sorted(per_emp.items(), key=lambda kv: kv[1], reverse=True)
        best_eid, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

        solo = (second_sim < 0)
        pass_thresh = best_sim >= THRESH
        pass_margin = True if solo else ((best_sim - second_sim) >= MARGIN)

        if not (pass_thresh and pass_margin):
            return Response({
                "error": "Face not recognized. Try again or re-enroll with more images.",
                "best_sim": best_sim,
                "second_sim": second_sim,
                # "meta": meta
            }, status=400)

        # Winner found → mark attendance (keep your original slot logic)
        emp = Employee.objects.get(id=best_eid)
        now = timezone.localtime()
        today = now.date()

        attendance, _created = Attendance.objects.get_or_create(
            user=emp, date=today, defaults={"status": "present"}
        )
        attendance.latitude = latitude
        attendance.longitude = longitude

        # Update slot timestamps
        if slot == 'office_in':
            attendance.check_in_time = now
        elif slot == 'break_in':
            attendance.break_in_time = now
        elif slot == 'break_out':
            attendance.break_out_time = now
        elif slot == 'office_out':
            attendance.check_out_time = now

        # Calculate late/early and save
        attendance.calculate_late_and_early()
        attendance.save()

        return Response({
            "status": "success",
            "message": f"Attendance marked for {emp.name} ({slot}).",
            "employee": {"id": emp.id, "name": emp.name, "email": emp.email},
            "time": now.strftime("%I:%M %p"),
            "confidence": best_sim,
            # "meta": meta  # includes detector score, blur, brightness, attempt etc.
        }, status=200)

# ------------------ Admin Login / Stats / Lists / Templates (unchanged) ------------------

class AdminLoginView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        if not email or not password:
            return Response({"error": "Email and password are required"}, status=400)
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Admin with this email does not exist."}, status=400)
        if not user.is_superuser:
            return Response({"error": "You must be an admin to login."}, status=403)
        user = authenticate(request, username=user.username, password=password)
        if user is not None:

            refresh = RefreshToken.for_user(user)
            return Response({"message": "Login successful", "access_token": str(refresh.access_token)}, status=200)
        return Response({"error": "Invalid email or password"}, status=400)

class AttendanceStatsView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        try:
            total_employees = Employee.objects.count()
            today = timezone.localdate()
            today_attendance_count = Attendance.objects.filter(date=today).count()
            return Response({"total_employees": total_employees, "today_attendance_count": today_attendance_count}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

@permission_classes([IsAdminUser])
class EmployeeListView(APIView):
    def get(self, request):
        employees = Employee.objects.all()
        serializer = EmployeeSerializer(employees, many=True, context={'request': request})
        return Response(serializer.data)

@login_required(login_url='admin-login')
def admin_login_view(request):
    if request.user.is_authenticated:
        return redirect('admin-dashboard')
    if request.method == 'POST':
        username = request.POST.get('username'); password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:
            login(request, user); return redirect('admin-dashboard')
        return render(request, 'login.html', {'error': 'Invalid credentials or not an admin user'})
    return render(request, 'login.html')

@login_required(login_url='admin-login')
def admin_dashboard_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    employees = Employee.objects.select_related('site', 'department').all()
    employees_by_site = defaultdict(list)
    for e in employees:
        site_name = e.site.name if e.site else None
        employees_by_site[site_name].append(e)
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
    if not request.user.is_staff:
        return redirect('admin-login')
    employee = get_object_or_404(Employee, id=user_id)
    filter_type = request.GET.get('filter', 'daily')
    today = timezone.now().date()
    SLOTS = {
        'Slot 1': {'time_range': '9:00 AM - 11:00 AM', 'slot_value': 'slot1'},
        'Slot 2': {'time_range': '11:00 AM - 1:00 PM', 'slot_value': 'slot2'},
        'Slot 3': {'time_range': '2:00 PM - 4:00 PM', 'slot_value': 'slot3'},
        'Slot 4': {'time_range': '4:00 PM - 6:00 PM', 'slot_value': 'slot4'},
    }
    context = {'employee': employee, 'filter': filter_type, 'today': today}
    if filter_type == 'daily':
        attendance_records = Attendance.objects.filter(user=employee, date=today)
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
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        attendance_records = Attendance.objects.filter(user=employee, date__range=[start_of_week, end_of_week])
        calendar_days = []
        current_date = start_of_week
        while current_date <= end_of_week:
            day_records = attendance_records.filter(date=current_date)
            day_slots = []
            for slot_name, slot_info in SLOTS.items():
                slot_record = day_records.filter(slot=slot_info['slot_value']).first()
                day_slots.append({
                    'name': slot_name.replace('Slot ', 'S'),
                    'status': slot_record.status if slot_record else 'empty'
                })
            calendar_days.append({'date': current_date, 'slots': day_slots})
            current_date += timedelta(days=1)
        context['calendar_days'] = calendar_days
        context['week_start'] = start_of_week
        context['week_end'] = end_of_week
        context['total_records'] = attendance_records.count()
        context['present_count'] = attendance_records.filter(status='present').count()
        context['late_count'] = attendance_records.filter(status='late').count()
        context['absent_count'] = attendance_records.filter(status='absent').count()
    else:
        attendance_records = Attendance.objects.filter(
            user=employee, date__month=today.month, date__year=today.year
        ).order_by('-date', 'slot')
        context['attendance_records'] = attendance_records
        context['total_records'] = attendance_records.count()
        context['present_count'] = attendance_records.filter(status='present').count()
        context['late_count'] = attendance_records.filter(status='late').count()
        context['absent_count'] = attendance_records.filter(status='absent').count()
    return render(request, 'user_detail.html', context)

@login_required(login_url='admin-login')
def admin_logout_view(request):
    logout(request)
    return redirect('admin-login')
