# attendance/views.py
import logging
import base64
import re
import xml.etree.ElementTree as ET
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
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import permission_classes

from django.contrib.auth.models import User
import numpy as np
from .models import Employee, Attendance, Site, FaceTemplate

# --- NEW: our engine/utils ---
from .engine import ENGINE
from .utils import (
    dataurl_to_bytes, pil_to_bgr_array_from_bytes,
    THRESH, MARGIN, get_image_bytes
)
from .serializers import *
from rest_framework_simplejwt.tokens import RefreshToken

logger = logging.getLogger(__name__)

# ------------------ Sites API ------------------


# ------------------ Sites API ------------------


class SiteListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        sites = Site.objects.all()
        serializer = SiteSerializer(sites, many=True)
        return Response(serializer.data)


class ImportEmployeesView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAdminUser]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file uploaded'}, status=400)

        try:
            import pandas as pd
            # Read the excel file
            # We'll read the first few rows to understand the structure
            df = pd.read_excel(file, header=None)
            
            # Locate header rows
            # Scan first 20 rows to find the header
            header_row_index = -1
            for i in range(20):
                row_values = df.iloc[i].astype(str).str.strip().tolist()
                # Check for key columns
                if any('Sr. Nr.' in val or 'Name' in val or 'Status' in val for val in row_values):
                    header_row_index = i
                    break
            
            if header_row_index == -1:
                 return Response({'error': 'Could not find header row (looked for "Sr. Nr.", "Name", or "Status")'}, status=400)

            header_row_1 = df.iloc[header_row_index].astype(str).str.strip().tolist()
            # Row 2 might be the next one, or merged. Let's assume next one for now if it exists
            header_row_2 = []
            if header_row_index + 1 < len(df):
                header_row_2 = df.iloc[header_row_index + 1].astype(str).str.strip().tolist()
            
            # Helper to find column index
            def find_col_index(keywords, row_list):
                for idx, val in enumerate(row_list):
                    if any(k.lower() in val.lower() for k in keywords):
                        return idx
                return -1

            # Map fields to column indices
            # We look in both rows because of merged headers
            col_map = {}
            
            # Basic Info
            col_map['name'] = find_col_index(['Name'], header_row_1)
            
            col_map['department'] = find_col_index(['Division'], header_row_1)
            
            col_map['position'] = find_col_index(['Designation'], header_row_1)
            
            col_map['badge_number'] = find_col_index(['KFD & KAMI', 'Emp. ID'], header_row_1)
            
            col_map['salary_grade'] = find_col_index(['Category'], header_row_1)
            
            col_map['nationality'] = find_col_index(['Nationality'], header_row_1)
            col_map['gender'] = find_col_index(['Gender'], header_row_1)
            col_map['marital_status'] = find_col_index(['Marital'], header_row_1)
            col_map['religion'] = find_col_index(['Religion'], header_row_1)
            col_map['visa_details'] = find_col_index(['Visa Details'], header_row_1)
            
            # Nested columns (Row 2) - usually under the main header
            # If header_row_2 is empty or useless, we might need to look at header_row_1 too or just rely on 2
            col_map['labor_card_number'] = find_col_index(['L.Card', 'CEC Nr'], header_row_2)
            if col_map['labor_card_number'] == -1: col_map['labor_card_number'] = find_col_index(['L.Card', 'CEC Nr'], header_row_1)

            col_map['mol_id'] = find_col_index(['Personal Nr'], header_row_2)
            if col_map['mol_id'] == -1: col_map['mol_id'] = find_col_index(['Personal Nr'], header_row_1)

            col_map['passport_number'] = find_col_index(['New Passport Nr', 'PP No'], header_row_2)
            if col_map['passport_number'] == -1: col_map['passport_number'] = find_col_index(['New Passport Nr', 'PP No'], header_row_1)

            col_map['passport_expiry'] = find_col_index(['Expiry Date'], header_row_2)
            
            # Dates
            col_map['dob'] = find_col_index(['Date of Birth'], header_row_1)
            col_map['doj'] = find_col_index(['D.O.J'], header_row_2)
            if col_map['doj'] == -1: col_map['doj'] = find_col_index(['D.O.J'], header_row_1)

            col_map['status'] = find_col_index(['Status'], header_row_1)
            col_map['site'] = find_col_index(['Project', 'Site'], header_row_1)

            # Process data starting from row AFTER headers
            # If we used header_row_index and header_row_index+1, data starts at header_row_index+2
            start_data_index = header_row_index + 2
            
            success_count = 0
            errors = []
            debug_info = []
            
            for index, row in df.iloc[start_data_index:].iterrows():
                try:
                    # Extract data using the map
                    def get_val(field):
                        idx = col_map.get(field)
                        if idx is not None and idx != -1:
                            val = row.iloc[idx]
                            return str(val).strip() if pd.notna(val) else None
                        return None

                    name = get_val('name')
                    if not name: 
                        debug_info.append(f"Row {index}: Skipped (No Name)")
                        continue # Skip empty rows

                    badge = get_val('badge_number')
                    # No auto-email generation
                    email = None
                    
                    # Check if exists based on badge or name
                    emp = None
                    if badge:
                        emp = Employee.objects.filter(badge_number=badge).first()
                    
                    if not emp:
                        emp = Employee.objects.filter(name=name).first()
                        
                    if not emp:
                        emp = Employee()

                    emp.name = name
                    # emp.email = email # Don't set email if it's None
                    emp.department = get_val('department')
                    emp.position = get_val('position')
                    emp.badge_number = badge
                    emp.salary_grade = get_val('salary_grade')
                    emp.nationality = get_val('nationality')
                    emp.gender = get_val('gender')
                    emp.marital_status = get_val('marital_status')
                    emp.religion = get_val('religion')
                    emp.visa_details = get_val('visa_details')
                    emp.labor_card_number = get_val('labor_card_number')
                    emp.mol_id = get_val('mol_id')
                    emp.passport_number = get_val('passport_number')
                    emp.status = get_val('status')
                    
                    # Handle Site
                    site_name = get_val('site')
                    if site_name:
                        site_obj = Site.objects.filter(name__iexact=site_name).first()
                        if not site_obj:
                            site_obj = Site.objects.create(name=site_name)
                        emp.site = site_obj
                    
                    # Handle Dates
                    def parse_date(date_str):
                        if not date_str: return None
                        try:
                            return pd.to_datetime(date_str).date()
                        except:
                            return None

                    emp.date_of_birth = parse_date(get_val('dob'))
                    emp.date_of_joining = parse_date(get_val('doj'))
                    emp.passport_expiry = parse_date(get_val('passport_expiry'))
                    
                    # Phone is required, use dummy if missing
                    if not emp.phone:
                        emp.phone = "0000000000"

                    emp.save()
                    success_count += 1
                    
                except Exception as e:
                    errors.append(f"Row {index}: {str(e)}")
            
            response_data = {
                'success': True, 
                'imported_count': success_count,
                'errors': errors[:10] # Return first 10 errors
            }
            
            if success_count == 0:
                response_data['debug'] = {
                    'col_map': col_map,
                    'header_row_1': header_row_1,
                    'header_row_2': header_row_2,
                    'row_logs': debug_info[:10]
                }
                
            return Response(response_data)

        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminAddEmployeeView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAdminUser]

    def post(self, request):
        data = request.data
        try:
            # Check if email exists
            email = data.get('email')
            name = data.get('name')
            badge = data.get('badge_number')
            
            # No auto-email generation
            
            # Check for duplicates based on badge or name if needed, but for manual add we might just allow it
            # or warn. User said "no any field is unique identifier".
            # So we just create.
            
            site_id = data.get('site')
            site = None
            if site_id:
                try:
                    site = Site.objects.get(id=site_id)
                except Site.DoesNotExist:
                    pass

            def parse_date(d): return d if d else None

            Employee.objects.create(
                name=name,
                email=email or None,
                phone=data.get('phone'),
                department=data.get('department'),
                position=data.get('position'),
                badge_number=badge,
                salary_grade=data.get('salary_grade'),
                status=data.get('status'),
                nationality=data.get('nationality'),
                gender=data.get('gender'),
                marital_status=data.get('marital_status'),
                religion=data.get('religion'),
                date_of_birth=parse_date(data.get('date_of_birth')),
                date_of_joining=parse_date(data.get('date_of_joining')),
                passport_number=data.get('passport_number'),
                passport_expiry=parse_date(data.get('passport_expiry')),
                visa_details=data.get('visa_details'),
                labor_card_number=data.get('labor_card_number'),
                mol_id=data.get('mol_id'),
                job_description=data.get('job_description'),
                employer=data.get('employer'),
                site=site
            )
            return Response({'success': True, 'message': 'Employee added successfully'})
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminEditEmployeeView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAdminUser]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
            data = {
                'id': emp.id,
                'name': emp.name,
                'email': emp.email,
                'phone': emp.phone,
                'department': emp.department,
                'position': emp.position,
                'badge_number': emp.badge_number,
                'salary_grade': emp.salary_grade,
                'status': emp.status,
                'nationality': emp.nationality,
                'gender': emp.gender,
                'marital_status': emp.marital_status,
                'religion': emp.religion,
                'date_of_birth': str(emp.date_of_birth) if emp.date_of_birth else '',
                'date_of_joining': str(emp.date_of_joining) if emp.date_of_joining else '',
                'passport_number': emp.passport_number,
                'passport_expiry': str(emp.passport_expiry) if emp.passport_expiry else '',
                'visa_details': emp.visa_details,
                'labor_card_number': emp.labor_card_number,
                'mol_id': emp.mol_id,
                'job_description': emp.job_description,
                'employer': emp.employer,
                'site': emp.site.id if emp.site else '',
            }
            return Response(data)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

    def put(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
            data = request.data
            
            emp.name = data.get('name', emp.name)
            emp.email = data.get('email') or None
            emp.phone = data.get('phone')
            emp.department = data.get('department')
            emp.position = data.get('position')
            emp.badge_number = data.get('badge_number')
            emp.salary_grade = data.get('salary_grade')
            emp.status = data.get('status')
            emp.nationality = data.get('nationality')
            emp.gender = data.get('gender')
            emp.marital_status = data.get('marital_status')
            emp.religion = data.get('religion')
            emp.passport_number = data.get('passport_number')
            emp.visa_details = data.get('visa_details')
            emp.labor_card_number = data.get('labor_card_number')
            emp.mol_id = data.get('mol_id')
            emp.job_description = data.get('job_description')
            emp.employer = data.get('employer')
            
            site_id = data.get('site')
            if site_id:
                try:
                    emp.site = Site.objects.get(id=site_id)
                except Site.DoesNotExist:
                    emp.site = None
            else:
                emp.site = None
            
            def parse_date(d): return d if d else None
            emp.date_of_birth = parse_date(data.get('date_of_birth'))
            emp.date_of_joining = parse_date(data.get('date_of_joining'))
            emp.passport_expiry = parse_date(data.get('passport_expiry'))
            
            emp.save()
            return Response({'success': True, 'message': 'Employee updated successfully'})
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminDeleteEmployeeView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAdminUser]

    def delete(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
            emp.delete()
            return Response({'success': True, 'message': 'Employee deleted successfully'})
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminBulkDeleteEmployeeView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAdminUser]

    def post(self, request):
        try:
            ids = request.data.get('ids', [])
            if not ids:
                return Response({'error': 'No IDs provided'}, status=400)
            
            Employee.objects.filter(id__in=ids).delete()
            return Response({'success': True, 'message': f'{len(ids)} employees deleted successfully'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)


# ------------------ Register User (DeepFace -> InsightFace) ------------------


class RegisterUserView(APIView):
    permission_classes = [AllowAny]

    @transaction.atomic
    def post(self, request):
        # Toggle this if you ever want to enforce strict quality.
        strict_mode = False  # lenient enrollment: accept any detected face

        # Validate text fields only
        s = EnrollSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        # Files MUST come from request.FILES
        files = request.FILES.getlist("images") or request.FILES.getlist("images[]")
        
        # DEBUG: Save incoming images to inspect them
        import os
        from datetime import datetime
        debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"[DEBUG] Received {len(files)} files")
        for idx, f in enumerate(files):
            debug_path = os.path.join(debug_dir, f'{timestamp}_image_{idx}_{f.name}')
            with open(debug_path, 'wb') as debug_file:
                f.seek(0)
                debug_file.write(f.read())
            print(f"[DEBUG] Saved image {idx}: {debug_path}")
            print(f"[DEBUG]   - Size: {f.size} bytes")
            print(f"[DEBUG]   - Content-Type: {f.content_type}")

        if not files:
            return Response(
                {
                    "error": "No images uploaded. Use form-data with key 'images' and attach files."
                },
                status=400,
            )

        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        department = data.get("department") or ""  # Changed to text field
        position = data.get("position") or ""
        job_description = data.get("job_description") or ""
        salary_grade = data.get("salary_grade") or ""
        badge_number = data.get("badge_number") or ""
        mol_id = data.get("mol_id") or ""
        labor_card_number = data.get("labor_card_number") or ""
        site_id = data.get("site")
        employer = data.get("employer") or ""

        if Employee.objects.filter(email=email).exists():
            return Response(
                {"error": "Employee with this email already exists."},
                status=400,
            )

        site_obj = Site.objects.filter(id=site_id).first() if site_id else None

        emp = Employee.objects.create(
            name=name,
            email=email,
            phone=phone,
            department=department,  # Now a simple text field
            position=position,
            job_description=job_description,
            salary_grade=salary_grade,
            badge_number=badge_number,
            mol_id=mol_id,
            labor_card_number=labor_card_number,
            site=site_obj,
            employer=employer,
        )

        valid_vecs = []
        rejected = []

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

                # Hard fail: no face / no embedding at all
                if v is None:
                    rejected.append(meta)
                    continue

                # Strict mode: also enforce quality gates
                if strict_mode and not meta.get("ok", False):
                    rejected.append(meta)
                    continue

                # Lenient mode: accept embedding even if meta["ok"] is False
                FaceTemplate.objects.create(
                    employee=emp,
                    embedding=v.tolist(),
                    quality=float(q),
                )
                valid_vecs.append(v)

            except Exception as e:  # noqa: BLE001
                rejected.append({"ok": False, "reason": str(e)})

        if not valid_vecs:
            emp.delete()
            return Response(
                {
                    "status": "error",
                    "message": "No valid faces detected in any uploaded images.",
                    "rejected": rejected,
                },
                status=422,
            )

        # Compute centroid and store profile pic
        centroid = np.mean(valid_vecs, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        emp.face_embedding = centroid.tolist()

        first_file = files[0]
        first_file.seek(0)
        emp.profile_picture.save(
            f"{emp.id}_profile_{first_file.name}",
            first_file,
            save=False,
        )
        emp.save(update_fields=["face_embedding", "profile_picture"])

        # Rebuild FAISS index
        qs = FaceTemplate.objects.all().only("id", "employee_id", "embedding")
        tuples = [
            (t.id, t.employee_id, np.array(t.embedding, dtype=np.float32))
            for t in qs
        ]
        ENGINE.rebuild_index(tuples)

        return Response(
            {
                "status": "success",
                "message": f"Employee '{emp.name}' enrolled successfully.",
                "templates_added": len(valid_vecs),
                "rejected": rejected,
                "employee": UserSerializer(emp, context={"request": request}).data,
            },
            status=201,
        )


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
        img = request.FILES.get("image") or request.FILES.get("image[]")
        if not img:
            return Response(
                {"error": "No image uploaded. Use form-data with key 'image'."},
                status=400,
            )

        # Read bytes exactly once
        img.seek(0)
        raw = img.read()
        if not raw:
            return Response(
                {"error": "Uploaded image is empty."},
                status=400,
            )

        # Convert to BGR ndarray and embed
        try:
            bgr = pil_to_bgr_array_from_bytes(raw)
        except Exception as e:  # noqa: BLE001
            return Response(
                {"error": f"Invalid image file. {e}"},
                status=400,
            )

        v, q, meta = ENGINE.embed_best_face(bgr)
        if v is None:
            return Response(
                {"error": "No face detected in image."},
                status=400,
            )

        # Quality gates (soft blur passes; others return actionable messages)
        if not meta.get("ok", False) and not str(
            meta.get("reason", ""),
        ).startswith("soft_blurry"):
            reason = str(meta.get("reason", ""))
            if "too_dark" in reason:
                return Response(
                    {"error": "Lighting too dim. Please brighten the environment."},
                    status=422,
                )
            if "too_bright" in reason:
                return Response(
                    {"error": "Image too bright. Avoid direct glare."},
                    status=422,
                )
            if "face_too_small" in reason:
                return Response(
                    {"error": "Move closer to the camera."},
                    status=422,
                )
            if "det_score" in reason or "blurry" in reason:
                return Response(
                    {"error": "Face not clear. Hold still and retry."},
                    status=422,
                )

        # Gallery must exist
        if ENGINE.index is None or len(ENGINE.ids) == 0:
            return Response(
                {"error": "No enrolled employees in gallery."},
                status=400,
            )

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
        print(f"Best employee ID: {best_eid}")
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

        solo = second_sim < 0
        pass_thresh = best_sim >= THRESH
        pass_margin = True if solo else (best_sim - second_sim) >= MARGIN

        if not (pass_thresh and pass_margin):
            return Response(
                {
                    "error": "Face not recognized. Try again or re-enroll with more images.",
                    "best_sim": best_sim,
                    "second_sim": second_sim,
                    # "meta": meta
                },
                status=400,
            )

        # Winner found → mark attendance (keep your original slot logic)
        emp = Employee.objects.get(id=best_eid)
        now = timezone.localtime()
        today = now.date()

        attendance, _created = Attendance.objects.get_or_create(
            user=emp,
            date=today,
            defaults={"status": "present"},
        )
        attendance.latitude = latitude
        attendance.longitude = longitude

        # Update slot timestamps
        if slot == "office_in":
            attendance.check_in_time = now
        elif slot == "break_in":
            attendance.break_in_time = now
        elif slot == "break_out":
            attendance.break_out_time = now
        elif slot == "office_out":
            attendance.check_out_time = now

        # Calculate late/early and save
        attendance.calculate_late_and_early()
        attendance.save()

        return Response(
            {
                "status": "success",
                "message": f"Attendance marked for {emp.name} ({slot}).",
                "employee": {"id": emp.id, "name": emp.name, "email": emp.email},
                "time": now.strftime("%I:%M %p"),
                "confidence": best_sim,
                # "meta": meta  # includes detector score, blur, brightness, attempt etc.
            },
            status=200,
        )


# ------------------ Admin Login / Stats / Lists / Templates (unchanged) ------------------


class AdminLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")
        if not email or not password:
            return Response(
                {"error": "Email and password are required"},
                status=400,
            )
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"error": "Admin with this email does not exist."},
                status=400,
            )
        if not user.is_superuser:
            return Response(
                {"error": "You must be an admin to login."},
                status=403,
            )
        user = authenticate(request, username=user.username, password=password)
        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "message": "Login successful",
                    "access_token": str(refresh.access_token),
                },
                status=200,
            )
        return Response({"error": "Invalid email or password"}, status=400)


class AttendanceStatsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            total_employees = Employee.objects.count()
            today = timezone.localdate()
            today_attendance_count = Attendance.objects.filter(date=today).count()
            return Response(
                {
                    "total_employees": total_employees,
                    "today_attendance_count": today_attendance_count,
                },
                status=200,
            )
        except Exception as e:  # noqa: BLE001
            return Response({"error": str(e)}, status=500)


@permission_classes([IsAdminUser])
class EmployeeListView(APIView):
    def get(self, request):
        employees = Employee.objects.all()
        serializer = EmployeeSerializer(
            employees,
            many=True,
            context={"request": request},
        )
        return Response(serializer.data)


# @login_required(login_url='admin-login')
def admin_login_view(request):
    if request.user.is_authenticated:
        return redirect("admin-dashboard")
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:
            login(request, user)
            return redirect("admin-dashboard")
        return render(
            request,
            "login.html",
            {"error": "Invalid credentials or not an admin user"},
        )
    return render(request, "login.html")


@login_required(login_url="admin-login")
def admin_dashboard_view(request):
    if not request.user.is_staff:
        return redirect("admin-login")
    
    # Get site filter from query params
    site_filter = request.GET.get('site', 'all')
    
    # Initialize employees queryset and selected_site
    employees = Employee.objects.select_related("site").order_by('name')
    selected_site = site_filter
    
    # Filter employees based on site selection
    if site_filter != 'all':
        try:
            selected_site_id = int(site_filter)
            employees = employees.filter(site_id=selected_site_id)
        except (ValueError, TypeError):
            # If site_filter is not a valid int, treat as 'all' or handle error
            pass # employees remains unfiltered or handle as needed
    
    # Pagination
    per_page = request.GET.get('per_page', 20)
    try:
        per_page = int(per_page)
        if per_page not in [20, 100, 500, 1000, 2000]:
            per_page = 20
    except ValueError:
        per_page = 20

    paginator = Paginator(employees, per_page)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    total_employees = Employee.objects.count()
    total_sites = Site.objects.count()
    today = timezone.now().date()
    today_attendance = Attendance.objects.filter(date=today).count()
    
    # Get all sites for the filter dropdown
    all_sites = Site.objects.all()
    # Serialize sites for JavaScript
    import json
    sites_json = json.dumps([{"id": site.id, "name": site.name} for site in all_sites])
    
    context = {
        "total_employees": total_employees,
        "total_sites": total_sites,
        "today_attendance": today_attendance,
        "all_sites": sites_json,
        "selected_site": site_filter,
        "employees": page_obj,
    }
    return render(request, "dashboard.html", context)


@login_required(login_url="admin-login")
def admin_user_detail_view(request, user_id):
    if not request.user.is_staff:
        return redirect("admin-login")
    employee = get_object_or_404(Employee, id=user_id)
    filter_type = request.GET.get("filter", "daily")
    today = timezone.now().date()
    SLOTS = {
        "Slot 1": {"time_range": "9:00 AM - 11:00 AM", "slot_value": "slot1"},
        "Slot 2": {"time_range": "11:00 AM - 1:00 PM", "slot_value": "slot2"},
        "Slot 3": {"time_range": "2:00 PM - 4:00 PM", "slot_value": "slot3"},
        "Slot 4": {"time_range": "4:00 PM - 6:00 PM", "slot_value": "slot4"},
    }
    context = {"employee": employee, "filter": filter_type, "today": today}
    if filter_type == "daily":
        attendance_records = Attendance.objects.filter(user=employee, date=today)
        slots_data = {}
        for slot_name, slot_info in SLOTS.items():
            slot_record = attendance_records.filter(
                slot=slot_info["slot_value"],
            ).first()
            slots_data[slot_name] = {
                "time_range": slot_info["time_range"],
                "status": slot_record.status if slot_record else None,
                "check_in": slot_record.check_in_time if slot_record else None,
                "late_minutes": slot_record.late_minutes if slot_record else 0,
                "latitude": slot_record.latitude if slot_record else None,
                "longitude": slot_record.longitude if slot_record else None,
            }
        context["slots"] = slots_data
        context["total_records"] = attendance_records.count()
        context["present_count"] = attendance_records.filter(
            status="present",
        ).count()
        context["late_count"] = attendance_records.filter(
            status="late",
        ).count()
        context["absent_count"] = attendance_records.filter(
            status="absent",
        ).count()
    elif filter_type == "weekly":
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        attendance_records = Attendance.objects.filter(
            user=employee,
            date__range=[start_of_week, end_of_week],
        )
        calendar_days = []
        current_date = start_of_week
        while current_date <= end_of_week:
            day_records = attendance_records.filter(date=current_date)
            day_slots = []
            for slot_name, slot_info in SLOTS.items():
                slot_record = day_records.filter(
                    slot=slot_info["slot_value"],
                ).first()
                day_slots.append(
                    {
                        "name": slot_name.replace("Slot ", "S"),
                        "status": slot_record.status if slot_record else "empty",
                    },
                )
            calendar_days.append(
                {
                    "date": current_date,
                    "slots": day_slots,
                },
            )
            current_date += timedelta(days=1)
        context["calendar_days"] = calendar_days
        context["week_start"] = start_of_week
        context["week_end"] = end_of_week
        context["total_records"] = attendance_records.count()
        context["present_count"] = attendance_records.filter(
            status="present",
        ).count()
        context["late_count"] = attendance_records.filter(
            status="late",
        ).count()
        context["absent_count"] = attendance_records.filter(
            status="absent",
        ).count()
    else:
        attendance_records = Attendance.objects.filter(
            user=employee,
            date__month=today.month,
            date__year=today.year,
        ).order_by("-date", "slot")
        context["attendance_records"] = attendance_records
        context["total_records"] = attendance_records.count()
        context["present_count"] = attendance_records.filter(
            status="present",
        ).count()
        context["late_count"] = attendance_records.filter(
            status="late",
        ).count()
        context["absent_count"] = attendance_records.filter(
            status="absent",
        ).count()
    return render(request, "user_detail.html", context)


@login_required(login_url="admin-login")
def admin_logout_view(request):
    logout(request)
    return redirect("admin-login")


# ------------------ Sites Management ------------------

@login_required(login_url="admin-login")
def admin_sites_view(request):
    """List all sites with employee counts"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    sites = Site.objects.all().order_by('name')
    sites_data = []
    for site in sites:
        employee_count = Employee.objects.filter(site=site).count()
        sites_data.append({
            'site': site,
            'employee_count': employee_count
        })
    
    # Pagination
    per_page = request.GET.get('per_page', 10)
    try:
        per_page = int(per_page)
    except ValueError:
        per_page = 10
        
    paginator = Paginator(sites_data, per_page)
    page = request.GET.get('page', 1)
    
    try:
        sites_page = paginator.page(page)
    except PageNotAnInteger:
        sites_page = paginator.page(1)
    except EmptyPage:
        sites_page = paginator.page(paginator.num_pages)
    
    context = {
        'sites_data': sites_page,
        'paginator': paginator,
        'per_page': per_page
    }
    return render(request, "sites.html", context)

class ImportSitesView(APIView):
    permission_classes = [IsAdminUser]
    authentication_classes = [SessionAuthentication]

    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        if not file.name.endswith(('.xlsx', '.xls')):
            return Response({'error': 'Invalid file format. Please upload Excel file.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            import pandas as pd
            df = pd.read_excel(file)
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Look for 'name' or 'site name' column
            name_col = next((col for col in df.columns if col in ['name', 'site name', 'site']), None)
            
            if not name_col:
                return Response({
                    'error': 'Column "Name" or "Site Name" not found.',
                    'debug': {'columns': list(df.columns)}
                }, status=status.HTTP_400_BAD_REQUEST)

            imported_count = 0
            errors = []

            for index, row in df.iterrows():
                try:
                    site_name = str(row[name_col]).strip()
                    if site_name and site_name.lower() != 'nan':
                        if not Site.objects.filter(name=site_name).exists():
                            Site.objects.create(name=site_name)
                            imported_count += 1
                except Exception as e:
                    errors.append(f"Row {index + 1}: {str(e)}")

            return Response({
                'message': 'Import successful',
                'imported_count': imported_count,
                'errors': errors
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@login_required(login_url="admin-login")
@require_http_methods(["GET", "POST"])
def admin_add_site(request):
    """Add a new site"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        if name:
            if Site.objects.filter(name=name).exists():
                return render(request, "sites.html", {
                    'error': f'Site "{name}" already exists.',
                    'sites_data': _get_sites_data()
                })
            
            coordinates = None
            if 'kml_file' in request.FILES:
                try:
                    kml_file = request.FILES['kml_file']
                    tree = ET.parse(kml_file)
                    root = tree.getroot()
                    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
                    coords_elements = root.findall('.//kml:coordinates', namespace)
                    
                    coords_list = []
                    for coord in coords_elements:
                        coords = coord.text.strip().split()
                        for point in coords:
                            parts = point.split(',')
                            if len(parts) >= 2:
                                lon, lat = parts[0], parts[1]
                                coords_list.append((float(lat), float(lon)))
                    
                    if coords_list:
                        coordinates = coords_list
                except Exception as e:
                    print(f"Error parsing KML: {e}")

            Site.objects.create(name=name, coordinates=coordinates)
            return redirect("admin-sites")
        else:
            return render(request, "sites.html", {
                'error': 'Site name is required.',
                'sites_data': _get_sites_data()
            })
    
    return redirect("admin-sites")


@login_required(login_url="admin-login")
@require_http_methods(["GET", "POST"])
def admin_edit_site(request, site_id):
    """Edit an existing site"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    site = get_object_or_404(Site, id=site_id)
    
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        if name:
            if Site.objects.filter(name=name).exclude(id=site_id).exists():
                return render(request, "sites.html", {
                    'error': f'Site "{name}" already exists.',
                    'sites_data': _get_sites_data()
                })
            
            # Handle KML File Update
            if 'kml_file' in request.FILES:
                try:
                    kml_file = request.FILES['kml_file']
                    tree = ET.parse(kml_file)
                    root = tree.getroot()
                    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
                    coords_elements = root.findall('.//kml:coordinates', namespace)
                    
                    coords_list = []
                    for coord in coords_elements:
                        coords = coord.text.strip().split()
                        for point in coords:
                            parts = point.split(',')
                            if len(parts) >= 2:
                                lon, lat = parts[0], parts[1]
                                coords_list.append((float(lat), float(lon)))
                    
                    if coords_list:
                        site.coordinates = coords_list
                except Exception as e:
                    print(f"Error parsing KML: {e}")
                    # Optionally handle error, e.g., return with error message
            
            site.name = name
            site.save()
            return redirect("admin-sites")
        else:
            return render(request, "sites.html", {
                'error': 'Site name is required.',
                'sites_data': _get_sites_data()
            })
    
    return redirect("admin-sites")


@login_required(login_url="admin-login")
@require_http_methods(["POST"])
def admin_delete_site(request, site_id):
    """Delete a site"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    site = get_object_or_404(Site, id=site_id)
    site.delete()
    return redirect("admin-sites")


@login_required(login_url="admin-login")
def admin_site_detail_view(request, site_id):
    """View site details and map"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    site = get_object_or_404(Site, id=site_id)
    employee_count = Employee.objects.filter(site=site).count()
    
    context = {
        'site': site,
        'employee_count': employee_count,
    }
    return render(request, "site_detail.html", context)


class SiteCoordinatesView(APIView):
    """API endpoint to get site coordinates for AJAX requests"""
    permission_classes = [AllowAny]
    
    def get(self, request, site_id):
        try:
            site = Site.objects.get(id=site_id)
            
            # Prepare coordinates for map
            map_coords = []
            center_lat = 25.0058  # Default center (Dubai)
            center_lng = 55.4364
            
            if site.coordinates:
                try:
                    # site.coordinates is stored as [(lat, lon), ...]
                    # We need [{'lat': lat, 'lng': lon}, ...]
                    for lat, lon in site.coordinates:
                        map_coords.append({'lat': float(lat), 'lng': float(lon)})
                    
                    # Calculate center if we have coordinates
                    if map_coords:
                        lats = [c['lat'] for c in map_coords]
                        lngs = [c['lng'] for c in map_coords]
                        center_lat = sum(lats) / len(lats)
                        center_lng = sum(lngs) / len(lngs)
                except Exception as e:
                    print(f"Error processing coordinates: {e}")
            
            return Response({
                'success': True,
                'coordinates': map_coords,
                'center': {
                    'lat': center_lat,
                    'lng': center_lng
                },
                'has_coordinates': len(map_coords) > 0
            }, status=200)
            
        except Site.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Site not found'
            }, status=404)


# Helper functions
def _get_sites_data():
    """Helper to get sites data with employee counts"""
    sites = Site.objects.all()
    sites_data = []
    for site in sites:
        employee_count = Employee.objects.filter(site=site).count()
        sites_data.append({
            'site': site,
            'employee_count': employee_count
        })
    return sites_data


class AdminBulkDeleteSiteView(APIView):
    permission_classes = [IsAdminUser]
    authentication_classes = [SessionAuthentication]

    def post(self, request):
        try:
            ids = request.data.get('ids', [])
            if not ids:
                return Response({'error': 'No IDs provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Delete sites
            deleted_count, _ = Site.objects.filter(id__in=ids).delete()

            return Response({
                'message': f'Successfully deleted {deleted_count} sites.'
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
