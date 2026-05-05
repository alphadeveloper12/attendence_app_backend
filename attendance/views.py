# attendance/views.py
import calendar
import pandas as pd
from fpdf import FPDF
from django.http import HttpResponse, FileResponse, JsonResponse
from django.template.loader import render_to_string
import io
import logging
import base64
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
from decimal import Decimal

from django.conf import settings
from django.db import transaction
from django.db.models import Count, Q, Sum
from django.utils import timezone
from django.core.files.base import ContentFile
from django.urls import reverse
from django.shortcuts import render, redirect, get_object_or_404

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import permission_classes

from django.contrib.auth.models import User
import numpy as np
from .models import Employee, Attendance, Site, FaceTemplate, AdminProfile

# --- NEW: our engine/utils ---
from .engine import ENGINE
from .utils import (
    dataurl_to_bytes, pil_to_bgr_array_from_bytes,
    THRESH, MARGIN, get_image_bytes
)
from .geofence import check_geofence
from .models import Employee, Attendance, Site, FaceTemplate, AdminProfile, AppBuild
from .serializers import *
from .permissions import IsSiteAdmin
from rest_framework_simplejwt.tokens import RefreshToken
from django.http import FileResponse, Http404

# ------------------ App Build API ------------------

class UploadBuildView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        if not request.user.is_superuser:
            return Response({'error': 'Only superusers can upload builds'}, status=403)
        
        app_type = request.data.get('app_type')
        file = request.FILES.get('file')
        version = request.data.get('version')

        if not app_type or not file:
            return Response({'error': 'App type and file are required'}, status=400)
        
        if app_type not in ['attendance', 'fuel']:
             return Response({'error': 'Invalid app type'}, status=400)

        try:
            # Delete existing build of same type
            existing_build = AppBuild.objects.filter(app_type=app_type).first()
            if existing_build:
                existing_build.file.delete()
                existing_build.delete()
            
            # Create new build
            build = AppBuild.objects.create(
                app_type=app_type,
                file=file,
                version=version
            )
            
            return Response({'message': f'{build.get_app_type_display()} uploaded successfully'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class DownloadBuildView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, app_type):
        if app_type not in ['attendance', 'fuel']:
            return Response({'error': 'Invalid app type'}, status=400)
            
        build = AppBuild.objects.filter(app_type=app_type).first()
        if not build or not build.file:
            raise Http404("Build not found")
            
        response = FileResponse(build.file.open('rb'), as_attachment=True)
        return response

logger = logging.getLogger(__name__)

# ── Simple in-memory rate limiter for 401 floods ─────────────────────────────
# Blocks a token (or IP) that causes > 10 consecutive 401s within 60 seconds.
import time as _time
from collections import defaultdict as _defaultdict
_rate_limit_store: dict = _defaultdict(lambda: {"count": 0, "first_seen": 0.0, "blocked_until": 0.0})
_RATE_LIMIT_MAX   = 10   # max consecutive 401s
_RATE_LIMIT_WINDOW = 60  # seconds before counter resets
_RATE_LIMIT_BLOCK  = 300 # block for 5 minutes after exceeding

def _check_rate_limit(key: str) -> bool:
    """Returns True if the key should be blocked."""
    now = _time.time()
    state = _rate_limit_store[key]
    if now < state["blocked_until"]:
        return True
    if now - state["first_seen"] > _RATE_LIMIT_WINDOW:
        state["count"] = 0
        state["first_seen"] = now
    state["count"] += 1
    if state["count"] > _RATE_LIMIT_MAX:
        state["blocked_until"] = now + _RATE_LIMIT_BLOCK
        logger.warning("[RATE-LIMIT] BLOCKED key=%r after %d hits", key, state["count"])
        return True
    return False

def _reset_rate_limit(key: str) -> None:
    """Call on successful auth to clear the counter."""
    if key in _rate_limit_store:
        _rate_limit_store[key] = {"count": 0, "first_seen": 0.0, "blocked_until": 0.0}
# ─────────────────────────────────────────────────────────────────────────────

# ------------------ Sites API ------------------


# ------------------ Sites API ------------------


class SiteListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        sites = Site.objects.all()
        serializer = SiteSerializer(sites, many=True)
        return Response(serializer.data)


class ImportEmployeesView(APIView):
    permission_classes = [IsAdminUser]

    @transaction.atomic
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file uploaded'}, status=400)

        try:
            # Read the excel file
            # We'll read the first few rows to understand the structure
            df = pd.read_excel(file, header=None, engine='openpyxl')
            
            # Locate header rows
            # Scan first 20 rows to find the header
            header_row_index = -1
            employer_name = None
            
            for i in range(20):
                row_values = df.iloc[i].astype(str).str.strip().tolist()
                
                # Try to find employer name in the first few rows
                if i < 5 and not employer_name:
                    for val in row_values:
                        if val and val.lower() != 'nan' and len(val) > 5:
                            employer_name = val
                            break

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
            
            col_map['job_description'] = find_col_index(['Present Designation'], header_row_1)

            # Refined badge number mapping to avoid 'Previous Emp. ID'
            col_map['badge_number'] = find_col_index(['Badge ID'], header_row_1)
            if col_map['badge_number'] == -1:
                # If not found by Badge ID, look for Emp. ID but exclude columns with 'Previous'
                for idx, val in enumerate(header_row_1):
                    if 'emp. id' in val.lower() and 'previous' not in val.lower():
                        col_map['badge_number'] = idx
                        break
            
            col_map['salary_grade'] = find_col_index(['Category'], header_row_1)
            
            col_map['gross_salary'] = find_col_index(['Gross Salary'], header_row_2)
            if col_map['gross_salary'] == -1: col_map['gross_salary'] = find_col_index(['Gross Salary'], header_row_1)
            
            col_map['basic_salary'] = find_col_index(['Basic Salary'], header_row_2)
            if col_map['basic_salary'] == -1: col_map['basic_salary'] = find_col_index(['Basic Salary'], header_row_1)

            col_map['nationality'] = find_col_index(['Nationality'], header_row_1)
            if col_map['nationality'] == -1: col_map['nationality'] = find_col_index(['Nationality'], header_row_2)

            col_map['gender'] = find_col_index(['Gender'], header_row_1)
            if col_map['gender'] == -1: col_map['gender'] = find_col_index(['Gender'], header_row_2)

            col_map['marital_status'] = find_col_index(['Marital'], header_row_1)
            if col_map['marital_status'] == -1: col_map['marital_status'] = find_col_index(['Marital'], header_row_2)

            col_map['religion'] = find_col_index(['Religion'], header_row_1)
            if col_map['religion'] == -1: col_map['religion'] = find_col_index(['Religion'], header_row_2)

            col_map['employer'] = find_col_index(['Employer'], header_row_1)
            if col_map['employer'] == -1: col_map['employer'] = find_col_index(['Employer'], header_row_2)

            col_map['visa_details'] = find_col_index(['Visa Detail', 'Visa Details'], header_row_1)
            if col_map['visa_details'] == -1: col_map['visa_details'] = find_col_index(['Visa Detail', 'Visa Details'], header_row_2)
            
            # Nested columns (Row 2) - usually under the main header
            # If header_row_2 is empty or useless, we might need to look at header_row_1 too or just rely on 2
            col_map['labor_card_number'] = find_col_index(['L.Card/CEC Nr', 'L.Card', 'CEC Nr'], header_row_2)
            if col_map['labor_card_number'] == -1: col_map['labor_card_number'] = find_col_index(['L.Card/CEC Nr', 'L.Card', 'CEC Nr'], header_row_1)

            col_map['mol_id'] = find_col_index(['Personal Nr'], header_row_2)
            if col_map['mol_id'] == -1: col_map['mol_id'] = find_col_index(['Personal Nr'], header_row_1)

            col_map['passport_number'] = find_col_index(['New Passport Nr', 'PP No'], header_row_2)
            if col_map['passport_number'] == -1: col_map['passport_number'] = find_col_index(['New Passport Nr', 'PP No'], header_row_1)

            col_map['passport_expiry'] = find_col_index(['Expiry Date'], header_row_2)
            if col_map['passport_expiry'] == -1: col_map['passport_expiry'] = find_col_index(['Expiry Date'], header_row_1)
            
            # Dates
            col_map['dob'] = find_col_index(['Date of Birth'], header_row_1)
            if col_map['dob'] == -1: col_map['dob'] = find_col_index(['Date of Birth'], header_row_2)

            col_map['doj'] = find_col_index(['D.O.J'], header_row_2)
            if col_map['doj'] == -1: col_map['doj'] = find_col_index(['D.O.J'], header_row_1)

            col_map['status'] = find_col_index(['Status'], header_row_1)
            col_map['site'] = find_col_index(['Project', 'Site'], header_row_1)

            # Process data starting from row AFTER headers
            start_data_index = header_row_index + 2
            
            # Helper to extract data from a specific row
            def get_val_from_row(row_data, field):
                idx = col_map.get(field)
                if idx is not None and idx != -1:
                    val = row_data.iloc[idx]
                    return str(val).strip() if pd.notna(val) else None
                return None

            success_count = 0
            errors = []
            
            # --- OPTIMIZATION: BATCH LOOKUPS ---
            # 1. Collect all potential identifiers from the sheet
            data_df = df.iloc[start_data_index:]
            all_badges = set()
            all_mol_ids = set()
            all_passports = set()
            all_site_names = set()

            for _, row in data_df.iterrows():
                b = get_val_from_row(row, 'badge_number')
                if b: all_badges.add(b)
                m = get_val_from_row(row, 'mol_id')
                if m: all_mol_ids.add(m)
                p = get_val_from_row(row, 'passport_number')
                if p: all_passports.add(p)
                s = get_val_from_row(row, 'site')
                if s: all_site_names.add(s.strip().lower())

            # 2. Bulk fetch existing employees and sites
            existing_emps_by_badge = {e.badge_number: e for e in Employee.objects.filter(badge_number__in=list(all_badges)).exclude(badge_number='') if e.badge_number}
            existing_emps_by_mol = {e.mol_id: e for e in Employee.objects.filter(mol_id__in=list(all_mol_ids)).exclude(mol_id='') if e.mol_id}
            existing_emps_by_passport = {e.passport_number: e for e in Employee.objects.filter(passport_number__in=list(all_passports)).exclude(passport_number='') if e.passport_number}
            
            site_cache = {s.name.lower(): s for s in Site.objects.all()}
            # -----------------------------------

            for index, row in data_df.iterrows():
                try:
                    name = get_val_from_row(row, 'name')
                    if not name: continue 

                    status = get_val_from_row(row, 'status')
                    badge = get_val_from_row(row, 'badge_number')
                    nationality = get_val_from_row(row, 'nationality')
                    
                    # Optimized lookup using memory cache
                    emp = existing_emps_by_badge.get(badge)
                    if not emp:
                        mol_id = get_val_from_row(row, 'mol_id')
                        emp = existing_emps_by_mol.get(mol_id)
                    if not emp:
                        passport_number = get_val_from_row(row, 'passport_number')
                        emp = existing_emps_by_passport.get(passport_number)
                    
                    if not emp and name and employer_name and nationality:
                        # Fallback for name-based lookup (rarer, keep as query for now or expand cache)
                        emp = Employee.objects.filter(name=name, employer=employer_name, nationality=nationality).first()
                    
                    is_new = not emp
                    if not emp:
                        emp = Employee()

                    if is_new:
                        emp.name = name
                        emp.department = get_val_from_row(row, 'department')
                        emp.position = get_val_from_row(row, 'position')
                        emp.badge_number = badge
                        emp.salary_grade = get_val_from_row(row, 'salary_grade')
                        emp.job_description = get_val_from_row(row, 'job_description')
                        emp.nationality = nationality
                        emp.gender = get_val_from_row(row, 'gender')
                        emp.marital_status = get_val_from_row(row, 'marital_status')
                        emp.religion = get_val_from_row(row, 'religion')
                        emp.labor_card_number = get_val_from_row(row, 'labor_card_number')
                        emp.mol_id = get_val_from_row(row, 'mol_id')
                        emp.passport_number = get_val_from_row(row, 'passport_number')
                        emp.status = status

                        # Salaries
                        def parse_float(val):
                            if not val: return None
                            try:
                                return float(str(val).replace(',', ''))
                            except:
                                return None

                        emp.gross_salary = parse_float(get_val_from_row(row, 'gross_salary'))
                        emp.basic_salary = parse_float(get_val_from_row(row, 'basic_salary'))
                        if not emp.basic_salary and emp.gross_salary:
                            emp.basic_salary = emp.gross_salary

                        # Category
                        div = (emp.department or "").lower()
                        cat = (emp.salary_grade or "").lower()
                        emp.category = 'staff' if 'staff' in div or 'staff' in cat or 'office' in div else 'worker'

                        # Handle Site using cache
                        site_name = get_val_from_row(row, 'site')
                        if site_name:
                            site_name = site_name.strip()
                            if site_name.upper() in ['HO', 'HEAD OFFICE']: site_name = 'Head Office'

                            site_key = site_name.lower()
                            if site_key in site_cache:
                                emp.site = site_cache[site_key]
                            else:
                                site_obj = Site.objects.create(name=site_name)
                                site_cache[site_key] = site_obj
                                emp.site = site_obj

                        # Handle Dates
                        def parse_date(date_str):
                            if not date_str: return None
                            try:
                                return pd.to_datetime(date_str).date()
                            except: return None

                        emp.date_of_birth = parse_date(get_val_from_row(row, 'dob'))
                        emp.date_of_joining = parse_date(get_val_from_row(row, 'doj'))
                        emp.passport_expiry = parse_date(get_val_from_row(row, 'passport_expiry'))

                        if not emp.phone: emp.phone = "0000000000"

                    # Always update employer and visa_details (for both new and existing employees)
                    emp.employer = get_val_from_row(row, 'employer') or employer_name
                    emp.visa_details = get_val_from_row(row, 'visa_details')

                    emp.save()
                    success_count += 1
                    
                except Exception as e:
                    errors.append(f"Row {index}: {str(e)}")
            
            return Response({'success': True, 'imported_count': success_count, 'errors': errors[:10]})

        except Exception as e:
            return Response({'error': str(e)}, status=500)


class DownloadEmployeeTemplateView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        import pandas as pd
        import io
        from django.http import HttpResponse

        # Create the multi-row header structure
        # Row 1: Employer Name (Placeholder)
        # Row 2: Empty
        # Row 3: Date (Placeholder)
        # Row 4: Main Headers
        # Row 5: Sub Headers (for merged columns)

        # Define headers based on the sample provided by the user
        header_row_1 = ["Sr. Nr.", "Status", "Status Date", "Category", "Division", "Project / Site", "Previous Emp. ID of PIC", "Badge ID", "Summary Code", "Name", "Present  Designation", "Nationality", "Gender", "Marital Status", "Religion", "Visa Details", "Labour Card Details", "", "Passport Details", "", "", "", "Current Salary Details", "", "", "", "", "", "", "", "Leave Entitlement Details", "Date of Birth (dd/mm/yyyy)"]
        header_row_2 = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Status", "L.Card/CEC Nr.", "Personal Nr.", "PP No. on Visa", "New Passport Nr.", "Expiry Date (dd/mm/yyyy)", "Basic Salary", "Accmn/CCA", "Transport/Special Allow", "Food Allowance", "Fixed OT allowance", " Others ", " Salary Reduction", " Gross Salary ", "D.O.J. (dd/mm/yyyy)", ""]

        # Sample data rows
        sample_row_1 = [1, "Active", "04/04/2024", "Worker", "Joinery", "Factory", "", "16227", "0002", "Hidayat Ullah Khan Rast Ali Khan", "Carpenter Finishing", "Pakistan", "Male", "Married", "Muslim", "PICDUB", "", "20001019369957", "ML4125622", "ML4125622", "", "02/04/2033", "800.00", "", "", "200.00", "", "450.00", "", "1,450.00", "05/01/2022", "01/01/1993"]
        sample_row_2 = [2, "Active", "15/08/2025", "Worker", "Joinery", "Lillia", "", "16620", "0002", "Lalchand Ram Keshav Ram", "Carpenter Finishing", "Indian", "Male", "Married", "Non Muslim", "PICDUB", "", "10001078361045", "M1264483", "X4403081", "", "30/07/2034", "1,000.00", "", "", "200.00", "", "200.00", "", "1,400.00", "08/04/2023", "01/07/1983"]
        sample_row_3 = [3, "leave", "14/02/2024", "Staff", "Joinery", "Park Horizon", "", "9249", "0002", "Mohammad Ahmad Mohammad Aaqil", "Foreman Painter", "Indian", "Male", "Married", "Muslim", "KPIC", "78888568", "10012076761949", "R7686385", "", "", "12/07/2027", "1,900.00", "", "", "", "", "1,400.00", "", "3,300.00", "05/08/2018", "12/07/1967"]

        # Create a DataFrame with the structure
        data = [
            header_row_1,
            header_row_2,
            [""] * len(header_row_1), # Empty row between headers and data
            sample_row_1,
            sample_row_2,
            sample_row_3
        ]

        df = pd.DataFrame(data)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, header=False, sheet_name='Employees')
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Employees']

            # Add some formatting
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            
            # Apply formatting to header rows (rows 1 and 2, 0-indexed: 0 and 1)
            for col_num, value in enumerate(header_row_1):
                worksheet.write(0, col_num, value, header_format)
            for col_num, value in enumerate(header_row_2):
                worksheet.write(1, col_num, value, header_format)

        output.seek(0)

        response = HttpResponse(
            output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=employee_import_template.xlsx'
        return response


class AdminAddEmployeeView(APIView):
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
            def parse_decimal(d):
                if not d or str(d).strip() == '':
                    return None
                try:
                    return Decimal(str(d).replace(',', ''))
                except:
                    return None

            status = data.get('status')
            resumption_date = parse_date(data.get('resumption_date'))
            # Note: when adding a new employee, no resumption date validation
            # (resumption is only relevant when an existing leave-employee returns to active)

            Employee.objects.create(
                name=name,
                email=email or None,
                phone=data.get('phone'),
                department=data.get('department'),
                position=data.get('position'),
                badge_number=badge,
                salary_grade=data.get('salary_grade'),
                status=status,
                resumption_date=resumption_date,
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
                site=site,
                gross_salary=parse_decimal(data.get('gross_salary')),
                basic_salary=parse_decimal(data.get('basic_salary')),
                category=data.get('category', 'worker'),
                camp=data.get('camp'),
                transportation=data.get('transportation')
            )
            return Response({'success': True, 'message': 'Employee added successfully'})
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminEditEmployeeView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]

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
                'resumption_date': str(emp.resumption_date) if emp.resumption_date else '',
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
                'camp': emp.camp,
                'transportation': emp.transportation,
            }
            if request.user.is_superuser:
                data['gross_salary'] = str(emp.gross_salary) if emp.gross_salary else ''
                data['basic_salary'] = str(emp.basic_salary) if emp.basic_salary else ''
                data['salary_grade'] = emp.salary_grade
            return Response(data)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

    def put(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
            data = request.data

            # Define helpers first — must be before any usage
            def parse_date(d): return d if d else None
            def parse_decimal(d):
                if not d or str(d).strip() == '':
                    return None
                try:
                    return Decimal(str(d).replace(',', ''))
                except:
                    return None

            # Resumption date logic:
            # Only required when transitioning Leave → Active
            # (i.e., when an employee on Leave is being marked as resumed)
            new_status     = data.get('status')
            new_resumption = parse_date(data.get('resumption_date'))
            old_status     = emp.status
            is_resuming    = (old_status == 'Leave' and new_status == 'Active')
            if is_resuming and not new_resumption:
                return Response(
                    {'error': 'Resumption date is required when bringing an employee back from Leave to Active.'},
                    status=400,
                )

            emp.name = data.get('name', emp.name)
            emp.email = data.get('email') or None
            emp.phone = data.get('phone')
            emp.department = data.get('department')
            emp.position = data.get('position')
            emp.badge_number = data.get('badge_number')
            emp.salary_grade = data.get('salary_grade')
            emp.status = new_status
            emp.resumption_date = new_resumption
            emp.nationality = data.get('nationality')
            emp.gender = data.get('gender')
            emp.marital_status = data.get('marital_status')
            emp.religion = data.get('religion')
            emp.passport_number = data.get('passport_number')
            emp.visa_details = data.get('visa_details')
            emp.labor_card_number = data.get('labor_card_number')
            emp.camp = data.get('camp')
            emp.transportation = data.get('transportation')
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

            emp.date_of_birth = parse_date(data.get('date_of_birth'))
            emp.date_of_joining = parse_date(data.get('date_of_joining'))
            emp.passport_expiry = parse_date(data.get('passport_expiry'))

            if request.user.is_superuser:
                emp.gross_salary = parse_decimal(data.get('gross_salary'))
                emp.basic_salary = parse_decimal(data.get('basic_salary'))
                emp.salary_grade = data.get('salary_grade', emp.salary_grade)

            emp.save()
            return Response({'success': True, 'message': 'Employee updated successfully'})
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AdminDeleteEmployeeView(APIView):
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
        if not s.is_valid():
            print(f"Serializer errors: {s.errors}")
            return Response(s.errors, status=400)
        data = s.validated_data
        print(request.data)

        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        department = data.get("department") or ""
        position = data.get("position") or ""
        job_description = data.get("job_description") or ""
        salary_grade = data.get("salary_grade") or ""
        badge_number = data.get("badge_number") or ""
        mol_id = data.get("mol_id") or ""
        labor_card_number = data.get("labor_card_number") or ""
        site_id = data.get("site")
        employer = data.get("employer") or ""
        nationality = data.get("nationality") or ""
        gender = data.get("gender") or ""
        marital_status = data.get("marital_status") or ""
        religion = data.get("religion") or ""
        date_of_birth = data.get("date_of_birth")
        date_of_joining = data.get("date_of_joining")
        passport_number = data.get("passport_number") or ""
        passport_expiry = data.get("passport_expiry")
        visa_details = data.get("visa_details") or ""
        status = data.get("status") or ""
        gross_salary = data.get("gross_salary")
        camp = data.get("camp") or ""
        transportation = data.get("transportation") or ""
        
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

        # Check if this is an edit or create operation
        employee_id = data.get("id")
        emp = None
        
        if employee_id:
            try:
                emp = Employee.objects.get(id=employee_id)
            except Employee.DoesNotExist:
                print(f"Employee with id {employee_id} not found")
                return Response({"error": "Employee not found"}, status=404)
        
        # If creating new user, check email uniqueness
        if not emp and email and Employee.objects.filter(email=email).exists():
            print(f"Employee with email {email} already exists")
            return Response(
                {"error": "Employee with this email already exists."},
                status=400,
            )

        # If creating new user, images are required
        if not emp and not files:
            print("No images uploaded for new user")
            return Response(
                {
                    "error": "No images uploaded. Use form-data with key 'images' and attach files."
                },
                status=400,
            )

        site_obj = Site.objects.filter(id=site_id).first() if site_id else None

        if emp:
            # Update existing employee
            emp.name = name
            emp.email = email
            emp.phone = phone
            emp.department = department
            emp.position = position
            emp.job_description = job_description
            emp.salary_grade = salary_grade
            emp.badge_number = badge_number
            emp.mol_id = mol_id
            emp.labor_card_number = labor_card_number
            emp.site = site_obj
            emp.employer = employer
            emp.nationality = nationality
            emp.gender = gender
            emp.marital_status = marital_status
            emp.religion = religion
            emp.date_of_birth = date_of_birth
            emp.date_of_joining = date_of_joining
            emp.passport_number = passport_number
            emp.passport_expiry = passport_expiry
            emp.visa_details = visa_details
            emp.status = status
            emp.gross_salary = gross_salary
            emp.camp = camp
            emp.transportation = transportation
            emp.save()
        else:
            # Create new employee
            emp = Employee.objects.create(
                name=name,
                email=email,
                phone=phone,
                department=department,
                position=position,
                job_description=job_description,
                salary_grade=salary_grade,
                badge_number=badge_number,
                mol_id=mol_id,
                labor_card_number=labor_card_number,
                site=site_obj,
                employer=employer,
                nationality=nationality,
                gender=gender,
                marital_status=marital_status,
                religion=religion,
                date_of_birth=date_of_birth,
                date_of_joining=date_of_joining,
                passport_number=passport_number,
                passport_expiry=passport_expiry,
                visa_details=visa_details,
                status=status,
                gross_salary=gross_salary,
                camp=camp,
                transportation=transportation,
            )

        # Process images if provided
        if files:
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
                # If new user and no valid faces, delete created user
                if not employee_id:
                    emp.delete()
                return Response(
                    {
                        "status": "error",
                        "message": "No valid faces detected in any uploaded images.",
                        "rejected": rejected,
                    },
                    status=422,
                )

            # Re-calculate centroid with ALL templates (old + new)
            all_templates = FaceTemplate.objects.filter(employee=emp)
            all_vecs = [np.array(t.embedding) for t in all_templates]
            
            if all_vecs:
                centroid = np.mean(all_vecs, axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                emp.face_embedding = centroid.tolist()

            # Update profile picture with the first new valid image
            first_file = files[0]
            first_file.seek(0)
            emp.profile_picture.save(
                f"{emp.id}_profile_{first_file.name}",
                first_file,
                save=False,
            )
            emp.save(update_fields=["face_embedding", "profile_picture"])

            # Rebuild FAISS index
            qs = FaceTemplate.objects.all().select_related('employee').only("id", "employee_id", "embedding", "employee__site_id")
            tuples = [
                (t.id, t.employee_id, t.employee.site_id, np.array(t.embedding, dtype=np.float32))
                for t in qs
            ]
            ENGINE.rebuild_index(tuples)

            return Response(
                {
                    "status": "success",
                    "message": f"Employee '{emp.name}' {'updated' if employee_id else 'enrolled'} successfully.",
                    "templates_added": len(valid_vecs),
                    "rejected": rejected,
                    "employee": UserSerializer(emp, context={"request": request}).data,
                },
                status=200,
            )
        
        # If no files provided (edit mode only), just return success
        return Response(
            {
                "status": "success",
                "message": f"Employee '{emp.name}' updated successfully.",
                "employee": UserSerializer(emp, context={"request": request}).data,
            },
            status=200,
        )


# ------------------ Mark Attendance (DeepFace -> InsightFace + FAISS) ------------------


import threading

def process_background_tasks(attendance_id, site_id, latitude, longitude):
    """
    Perform heavy calculations in background:
    - Geofence check
    - Late/Early minutes calculation
    - Final save
    """
    try:
        # Re-fetch fresh object
        att = Attendance.objects.get(id=attendance_id)
        
        # 1. Geofence Check
        site = None
        if site_id:
            try:
                site = Site.objects.get(id=site_id)
            except Site.DoesNotExist:
                pass
        
        # Fallback to employee site
        if not site and att.user.site:
            site = att.user.site

        is_within = check_geofence(site, latitude, longitude)
        att.is_within_geofence = is_within

        # 2. Calculate Late/Early (Pure math now, no save)
        att.calculate_late_and_early()
        
        # 3. Final Save
        att.save()
        
    except Exception as e:
        print(f"Background task failed for attendance {attendance_id}: {e}")

class MarkAttendanceView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # ── DEBUG: log every incoming request ──────────────────────────────
        req_employee_id = request.data.get("employee_id", "NONE")
        req_slot        = request.data.get("slot", "NONE")
        req_site_id     = request.data.get("site_id", "NONE")
        req_timestamp   = request.data.get("timestamp", "NONE")
        req_lat         = request.data.get("latitude", "NONE")
        req_lng         = request.data.get("longitude", "NONE")
        req_has_image   = bool(request.FILES.get("image") or request.FILES.get("image[]"))
        logger.info(
            "[MARK-ATTENDANCE] INCOMING | employee_id=%r slot=%r site=%r "
            "timestamp=%r lat=%r lng=%r has_image=%r",
            req_employee_id, req_slot, req_site_id,
            req_timestamp, req_lat, req_lng, req_has_image,
        )
        # ────────────────────────────────────────────────────────────────────

        # Validate non-file fields first (slot/lat/long)
        s = VerifySerializer(data=request.data)
        if not s.is_valid():
            logger.warning("[MARK-ATTENDANCE] SERIALIZER FAIL | errors=%r | data=%r", s.errors, request.data)
            raise Exception(s.errors)
        data = s.validated_data

        slot = data["slot"]
        latitude = data["latitude"]
        longitude = data["longitude"]

        if slot not in ["office_in", "office_out"]:
            logger.warning("[MARK-ATTENDANCE] INVALID SLOT | slot=%r", slot)
            return Response(
                {"error": "Invalid slot. Must be 'office_in' or 'office_out'."},
                status=400,
            )

        # File MUST come from request.FILES (avoid serializer coercion)
        img = request.FILES.get("image") or request.FILES.get("image[]")

        # Extract employee_id early — offline sync already matched on device
        employee_id_input = data.get("employee_id")

        # Image is optional when employee_id is provided (offline sync with deleted cache)
        if not img:
            if employee_id_input:
                # Offline sync — face already matched on device, skip all image processing
                logger.info(
                    "mark-attendance | no_image offline_sync | employee_id=%r | site_id=%r | slot=%r",
                    employee_id_input, data.get("site_id"), slot,
                )
                bgr = None
                raw = None
                v = None
                q = 0.0
                meta = {"ok": True, "reason": "offline_no_image"}
            else:
                return Response(
                    {"error": "No image uploaded. Use form-data with key 'image'."},
                    status=400,
                )
        else:
            # Read bytes exactly once
            img.seek(0)
            raw = img.read()
            if not raw:
                if employee_id_input:
                    bgr = None
                    v = None
                    q = 0.0
                    meta = {"ok": True, "reason": "offline_empty_image"}
                else:
                    return Response({"error": "Uploaded image is empty."}, status=400)
            else:
                # Convert to BGR ndarray and embed
                try:
                    bgr = pil_to_bgr_array_from_bytes(raw)
                except Exception as e:  # noqa: BLE001
                    if employee_id_input:
                        bgr = None
                        v = None
                        q = 0.0
                        meta = {"ok": True, "reason": "offline_invalid_image"}
                    else:
                        return Response({"error": f"Invalid image file. {e}"}, status=400)
                else:
                    v, _q, meta = ENGINE.embed_best_face(bgr)
        if v is None:
            if employee_id_input:
                # Offline sync: face match already done on device, no embedding needed
                logger.info(
                    "mark-attendance | no_face_detected but employee_id provided "
                    "(offline sync) | employee_id=%r | site_id=%r | slot=%r",
                    employee_id_input,
                    data.get("site_id"),
                    slot,
                )
                v = None  # will fall through to offline fallback below
            else:
                return Response(
                    {"error": "No face detected in image."},
                    status=400,
                )

        # Quality gates — skipped for offline sync (employee already matched on device)
        if v is not None and not employee_id_input:
            if not meta.get("ok", False) and not str(
                meta.get("reason", ""),
            ).startswith("soft_blurry"):
                reason = str(meta.get("reason", ""))
                logger.warning(
                    "mark-attendance 422 | quality_gate_failed | reason=%r | meta=%s | "
                    "site_id=%r | slot=%r",
                    reason,
                    meta,
                    data.get("site_id"),
                    slot,
                )
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
                # reason didn't match any known pattern — log and fall through
                logger.warning(
                    "mark-attendance 422 | unmatched_quality_reason=%r | meta=%s",
                    reason,
                    meta,
                )

        # Gallery must exist (skip check for offline sync — not using gallery)
        if not employee_id_input and ENGINE.indices == {} and not ENGINE.ids:
             return Response(
                {"error": "No enrolled employees in gallery."},
                status=400,
            )

        site_id = data.get("site_id")
        # employee_id_input already assigned above (before quality gate)

        best_eid = None
        best_sim = 0.0
        second_sim = -1.0

        if v is None and employee_id_input:
            # Offline sync: face undetectable but match already done on device — trust it
            logger.info(
                "mark-attendance | offline_sync_no_face | employee_id=%r | site_id=%r | slot=%r",
                employee_id_input, site_id, slot,
            )
            best_eid = employee_id_input
            best_sim = 1.0
        else:
            # FAISS nearest neighbors (cosine similarity on L2-normalized vectors)
            results = ENGINE.search(v, k=10, site_id=site_id)

        if best_eid is None:
            # Run FAISS recognition (live path — v is guaranteed non-None here)
            if results:
                # Aggregate to best per employee
                per_emp = {}
                for _, eid, sim in results:
                    if eid not in per_emp or sim > per_emp[eid]:
                        per_emp[eid] = sim

                # Decide winner with threshold + margin
                ranked = sorted(per_emp.items(), key=lambda kv: kv[1], reverse=True)
                best_eid, best_sim = ranked[0]
                second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

                solo = second_sim < 0
                pass_thresh = best_sim >= THRESH
                pass_margin = True if solo else (best_sim - second_sim) >= MARGIN

                if not (pass_thresh and pass_margin):
                    print("Face recognition failed: Threshold or Margin not met")
                    # IF we have an employee_id_input (from offline sync), we TRUST it
                    if employee_id_input:
                        print(f"Trusting offline identification: {employee_id_input}")
                        best_eid = employee_id_input
                    else:
                        return Response(
                            {
                                "error": "Face not recognized. Try again or re-enroll with more images.",
                                "best_sim": best_sim,
                                "second_sim": second_sim,
                            },
                            status=400,
                        )
            else:
                # No results from engine search
                if employee_id_input:
                    print(f"No match found in engine, but using provided employee_id: {employee_id_input}")
                    best_eid = employee_id_input
                else:
                    return Response({"error": "No match found."}, status=400)

        # Winner found → mark attendance
        try:
            emp = Employee.objects.get(id=best_eid)
            logger.info("[MARK-ATTENDANCE] EMPLOYEE FOUND | id=%r name=%r site=%r", emp.id, emp.name, emp.site_id)
        except Employee.DoesNotExist:
            logger.error("[MARK-ATTENDANCE] EMPLOYEE NOT FOUND | best_eid=%r employee_id_input=%r", best_eid, employee_id_input)
            return Response({"error": f"Employee {best_eid} not found."}, status=400)

        # Parse and convert to local time explicitly
        provided_timestamp = data.get("timestamp")
        if provided_timestamp:
            now = timezone.localtime(provided_timestamp)
            logger.info("[MARK-ATTENDANCE] TIMESTAMP | provided=%r → local=%r", provided_timestamp, now)
        else:
            now = timezone.localtime()
            logger.info("[MARK-ATTENDANCE] TIMESTAMP | none provided, using now=%r", now)

        today = now.date()

        # Check for existing attendance for today
        attendance, created = Attendance.objects.get_or_create(
            user=emp,
            date=today,
            defaults={"status": "present"},
        )
        logger.info(
            "[MARK-ATTENDANCE] ATTENDANCE RECORD | emp=%r date=%r created=%r "
            "check_in=%r check_out=%r",
            emp.id, today, created,
            attendance.check_in_time, attendance.check_out_time,
        )

        # Prevent duplicate markings for the same slot
        if slot == "office_in" and attendance.check_in_time and not created:
            local_time_str = timezone.localtime(attendance.check_in_time).strftime('%I:%M %p')
            logger.warning(
                "[MARK-ATTENDANCE] DUPLICATE | emp=%r slot=office_in already=%r",
                emp.id, local_time_str,
            )
            return Response(
                {"error": f"Attendance 'office_in' already marked for {emp.name} today at {local_time_str}."},
                status=400,
            )
        if slot == "office_out" and attendance.check_out_time and not created:
            local_time_str = timezone.localtime(attendance.check_out_time).strftime('%I:%M %p')
            logger.warning(
                "[MARK-ATTENDANCE] DUPLICATE | emp=%r slot=office_out already=%r",
                emp.id, local_time_str,
            )
            return Response(
                {"error": f"Attendance 'office_out' already marked for {emp.name} today at {local_time_str}."},
                status=400,
            )

        # Update basic fields IMMEDIATELY
        attendance.latitude = latitude
        attendance.longitude = longitude
        attendance.slot = slot

        # Update slot timestamps
        if slot == "office_in":
            attendance.check_in_time = now
        elif slot == "office_out":
            attendance.check_out_time = now

        # Basic Save (Fast)
        try:
            attendance.save()
            logger.info(
                "[MARK-ATTENDANCE] SUCCESS | emp=%r name=%r slot=%r time=%r",
                emp.id, emp.name, slot, now.strftime("%I:%M %p"),
            )
        except Exception as db_err:
            logger.error("[MARK-ATTENDANCE] DB SAVE FAILED | emp=%r slot=%r error=%r", emp.id, slot, str(db_err))
            return Response({"error": f"Database save failed: {db_err}"}, status=500)

        # Offload Heavy Logic (Geofence + Late/Early Calc) to Background Thread
        t = threading.Thread(
            target=process_background_tasks,
            args=(attendance.id, site_id, latitude, longitude)
        )
        t.start()

        return Response(
            {
                "status": "success",
                "message": f"Attendance marked for {emp.name} ({slot}).",
                "employee": {"id": emp.id, "name": emp.name, "email": emp.email},
                "time": now.strftime("%I:%M %p"),
                "confidence": best_sim,
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
            # Check if site admin
            if not AdminProfile.objects.filter(user=user).exists():
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
                    "is_superuser": user.is_superuser,
                },
                status=200,
            )
        return Response({"error": "Invalid email or password"}, status=400)


class AttendanceStatsView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]
    
    def get(self, request):
        try:
            employees = Employee.objects.all()
            attendance = Attendance.objects.filter(date=timezone.localdate())
            all_sites = Site.objects.all()

            # Filter by Site/Category (Query Param or Admin Profile)
            site_id = request.GET.get('site')
            category_filter = request.GET.get('category')
            status_filter = request.GET.get('status')
            
            if not request.user.is_superuser:
                try:
                    profile = AdminProfile.objects.get(user=request.user)
                    assigned_sites = profile.sites.all()
                    all_sites = assigned_sites
                    
                    if site_id and site_id != 'all':
                        if not assigned_sites.filter(id=site_id).exists():
                            employees = employees.none()
                            attendance = attendance.none()
                        else:
                            employees = employees.filter(site_id=site_id)
                            attendance = attendance.filter(user__site_id=site_id)
                    else:
                        employees = employees.filter(site__in=assigned_sites)
                        attendance = attendance.filter(user__site__in=assigned_sites)
                except AdminProfile.DoesNotExist:
                    pass
            elif site_id and site_id != 'all':
                employees = employees.filter(site_id=site_id)
                attendance = attendance.filter(user__site_id=site_id)

            if category_filter and category_filter != 'all':
                employees = employees.filter(
                    Q(salary_grade__iexact=category_filter) | 
                    (Q(salary_grade__in=['', None]) & Q(category__iexact=category_filter))
                )
                attendance = attendance.filter(
                    Q(user__salary_grade__iexact=category_filter) | 
                    (Q(user__salary_grade__in=['', None]) & Q(user__category__iexact=category_filter))
                )

            if status_filter and status_filter != 'all':
                employees = employees.filter(status__iexact=status_filter)
                attendance = attendance.filter(user__status__iexact=status_filter)

            # Unique categories from both category and salary_grade
            # Get union of both fields, strip, and unify case-insensitively
            cats = set(Employee.objects.exclude(category__isnull=True).exclude(category='').values_list('category', flat=True))
            grades = set(Employee.objects.exclude(salary_grade__isnull=True).exclude(salary_grade='').values_list('salary_grade', flat=True))
            
            unified_cats = {}
            for c in (list(cats) + list(grades)):
                if not c or not c.strip(): continue
                c_strip = c.strip()
                c_lower = c_strip.lower()
                if c_lower not in unified_cats:
                    unified_cats[c_lower] = c_strip.capitalize() # Standardize to Title Case
            
            unique_categories = sorted(list(unified_cats.values()))

            # Master list — always present regardless of whether any employee has that status
            MASTER_STATUSES = ['Active', 'Leave', 'Offboarded', 'Resigned', 'No Renewal', 'Terminated', 'Other']
            db_statuses = Employee.objects.exclude(status__isnull=True).exclude(status='').values_list('status', flat=True).distinct()
            # Merge: master list first, then any custom values added via import
            merged = list(dict.fromkeys(MASTER_STATUSES + sorted(set(db_statuses) - set(MASTER_STATUSES))))
            unique_statuses = merged

            # Chart Data (Last 7 Days)
            today = timezone.localdate()
            dates = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
            chart_labels = [d.strftime("%a") for d in dates]
            chart_data = []
            
            # Base query for chart (depends on site filter)
            base_qs = Attendance.objects.all()
            if site_id and site_id != 'all':
                base_qs = base_qs.filter(user__site_id=site_id)
            
            for d in dates:
                count = base_qs.filter(date=d).count()
                chart_data.append(count)

            return Response(
                {
                    "total_employees": employees.count(),
                    "today_attendance_count": attendance.count(),
                    "late_count": attendance.filter(status='late').count(),
                    "total_sites": all_sites.count(),
                    "sites": [{"id": s.id, "name": s.name} for s in all_sites],
                    "categories": unique_categories,
                    "statuses": unique_statuses,
                    "chart": {
                        "labels": chart_labels,
                        "data": chart_data
                    }
                },
                status=200,
            )
        except Exception as e:  # noqa: BLE001
            return Response({"error": str(e)}, status=500)


class AttendanceAlertsView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]
    
    def get(self, request):
        today = timezone.localdate()
        alerts = Attendance.objects.filter(
            date=today, 
            is_within_geofence=False
        ).select_related('user', 'user__site')

        # Filter
        site_id = request.GET.get('site')
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id and site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                        alerts = alerts.none()
                    else:
                        alerts = alerts.filter(user__site_id=site_id)
                else:
                    alerts = alerts.filter(user__site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                alerts = alerts.none()
        elif site_id and site_id != 'all':
            alerts = alerts.filter(user__site_id=site_id)

        data = []
        for a in alerts:
            data.append({
                "id": a.id,
                "user_name": a.user.name,
                "user_id": a.user.id,
                "user_pic": a.user.profile_picture.url if a.user.profile_picture else None,
                "site": a.user.site.name if a.user.site else "-",
                "time": a.check_in_time.strftime("%H:%M") if a.check_in_time else (a.check_out_time.strftime("%H:%M") if a.check_out_time else "-"),
                "lat": a.latitude,
                "long": a.longitude,
                "status": "Out of Bounds"
            })
        return Response(data)


@permission_classes([IsAdminUser | IsSiteAdmin])
class EmployeeListView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]
    
    def get(self, request):
        employees = Employee.objects.select_related('site').all().order_by('name')
        
        # Filter by face embedding presence
        has_face = request.GET.get('has_face')
        if has_face == 'true':
            # Exclude null, empty strings, and empty lists from JSONField
            employees = employees.filter(face_embedding__isnull=False).exclude(face_embedding="").exclude(face_embedding=[])
        
        # Filter by site
        site_id = request.GET.get('site')
        if not request.user.is_superuser and request.user.is_authenticated: # Keep original condition for site admin check
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id and site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                        return Response({"results": [], "count": 0})
                    employees = employees.filter(site_id=site_id)
                else:
                    employees = employees.filter(site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                return Response({"results": [], "count": 0})
        elif site_id and site_id != 'all':
            employees = employees.filter(site_id=site_id)
        elif not request.user.is_authenticated and not request.GET.get('site'):
            # For anonymous access (debugging), allow fetching all if no site is specified
            pass

        if site_id and site_id != 'all':
            employees = employees.filter(site_id=site_id)

        # Filter by category
        category_filter = request.GET.get('category')

        if category_filter and category_filter != 'all':
            employees = employees.filter(
                Q(salary_grade__iexact=category_filter) | 
                (Q(salary_grade__in=['', None]) & Q(category__iexact=category_filter))
            )

        # Filter by status
        status_filter = request.GET.get('status')
        if status_filter and status_filter != 'all':
            employees = employees.filter(status__iexact=status_filter)

        # Search
        search = request.GET.get('search')
        if search:
            employees = employees.filter(
                Q(name__icontains=search) | 
                Q(email__icontains=search) | 
                Q(badge_number__icontains=search)
            )

        # Pagination
        paginator = PageNumberPagination()
        # Default to a large number if not specified, but dashboard specifically sends per_page
        paginator.page_size = int(request.GET.get('per_page', 1000))

        # Attendance Filter (Present/Late) - Apply after other filters but before pagination
        attendance_filter = request.GET.get('attendance_filter')
        if attendance_filter in ['present', 'late']:
            today = timezone.localdate()
            attendance_qs = Attendance.objects.filter(date=today)
            if attendance_filter == 'late':
                attendance_qs = attendance_qs.filter(status='late')
            else:
                attendance_qs = attendance_qs.filter(status='present')
            
            present_employee_ids = attendance_qs.values_list('user_id', flat=True)
            employees = employees.filter(id__in=present_employee_ids)

        result_page = paginator.paginate_queryset(employees, request)
        
        serializer = EmployeeSerializer(
            result_page,
            many=True,
            context={"request": request},
        )
        return paginator.get_paginated_response(serializer.data)


# @login_required(login_url='admin-login')
def admin_login_view(request):
    if request.user.is_authenticated:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'redirect_url': reverse('admin-dashboard')})
        return redirect("admin-dashboard")
        
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect_url': reverse('admin-dashboard')})
            return redirect("admin-dashboard")
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': 'Invalid credentials or not an admin user'})
            
        return render(
            request,
            "login.html",
            {
                "error": "Invalid credentials or not an admin user",
                "hide_sidebar": True
            },
        )
    return render(request, "login.html", {"hide_sidebar": True})


def admin_downloads_view(request):
    """Publicly accessible downloads page for app builds"""
    builds = AppBuild.objects.all().order_by('-uploaded_at')
    return render(request, "downloads.html", {
        "builds": builds,
        "hide_sidebar": True
    })


@login_required(login_url="admin-login")
def admin_dashboard_view(request):
    if not request.user.is_staff:
        return redirect("admin-login")
    
    # Check if user is a Site Admin
    is_superuser = request.user.is_superuser
    site_admin_profile = None
    if not is_superuser:
        try:
            site_admin_profile = AdminProfile.objects.get(user=request.user)
        except AdminProfile.DoesNotExist:
            pass
            
    # Get site filter from query params
    site_filter = request.GET.get('site', 'all')
    
    # Initialize employees queryset
    employees = Employee.objects.select_related("site").order_by('name')
    selected_site = site_filter
    
    # Role-based filtering
    if not is_superuser and site_admin_profile:
        assigned_sites = site_admin_profile.sites.all()
        if site_filter != 'all':
            try:
                selected_site_id = int(site_filter)
                if not assigned_sites.filter(id=selected_site_id).exists():
                    employees = employees.none()
                else:
                    employees = employees.filter(site_id=selected_site_id)
                    selected_site = site_filter
            except (ValueError, TypeError):
                pass
        else:
            employees = employees.filter(site__in=assigned_sites)
            selected_site = 'all'
    elif site_filter != 'all':
        try:
            selected_site_id = int(site_filter)
            employees = employees.filter(site_id=selected_site_id)
        except (ValueError, TypeError):
            pass
            
    # Status Filtering
    status_filter = request.GET.get('status', 'all')
    if status_filter != 'all':
        employees = employees.filter(status__iexact=status_filter)
            
    # Search Filtering
    search_query = request.GET.get('search', '')
    if search_query:
        employees = employees.filter(
            Q(name__icontains=search_query) | 
            Q(email__icontains=search_query) | 
            Q(badge_number__icontains=search_query)
        )
    
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
    today = timezone.localdate()
    today_attendance = Attendance.objects.filter(date=today).count()
    
    # Get all sites for the filter dropdown
    if not is_superuser and site_admin_profile:
        all_sites = site_admin_profile.sites.all()
    else:
        all_sites = Site.objects.all()
    # Serialize sites for JavaScript
    import json
    sites_json = json.dumps([{"id": site.id, "name": site.name} for site in all_sites])
    
    # Geofence Alerts (Today)
    # Recalculate geofence status for all of today's records to ensure accuracy with latest site boundaries
    today_records = Attendance.objects.filter(date=today).select_related('user', 'user__site')
    for record in today_records:
        if record.latitude and record.longitude:
            # Determine site: use record's user site, or if not set, maybe a default?
            # Logic in MarkAttendanceView: if site_id provided use it, else user.site
            # Here we only have user.site easily accessible. 
            # If the user was at a different site, we might not know which one unless we stored it.
            # But typically employees are checked against their assigned site.
            site_to_check = record.user.site
            
            if site_to_check:
                is_within = check_geofence(site_to_check, record.latitude, record.longitude)
                if is_within != record.is_within_geofence:
                    record.is_within_geofence = is_within
                    record.save(update_fields=['is_within_geofence'])

    geofence_alerts = Attendance.objects.filter(
        date=today, 
        is_within_geofence=False
    ).select_related('user', 'user__site')
    
    if not is_superuser and site_admin_profile:
        geofence_alerts = geofence_alerts.filter(user__site__in=site_admin_profile.sites.all())
    
    context = {
        "total_employees": total_employees, # This might need to be filtered too for site admin? 
                                            # User said "can see his site employees only". 
                                            # So total_employees stats should probably reflect that?
                                            # But usually "Total Employees" on dashboard means global.
                                            # Let's keep it global for stats, but list is filtered.
                                            # Actually, if I am site admin, I probably only care about my site stats.
                                            # Let's filter stats if site admin.
        "total_sites": total_sites,
        "today_attendance": today_attendance, # Should also be filtered?
        "all_sites": sites_json,
        "selected_site": site_filter,
        "all_sites": sites_json,
        "selected_site": site_filter,
        "employees": page_obj,
        "search_query": search_query,
        "geofence_alerts": geofence_alerts,
        "is_superuser": is_superuser,
        "site_admin_sites": site_admin_profile.sites.all() if site_admin_profile else None
    }
    
    if not is_superuser and site_admin_profile:
        assigned_sites = site_admin_profile.sites.all()
        context["total_employees"] = Employee.objects.filter(site__in=assigned_sites).count()
        context["today_attendance"] = Attendance.objects.filter(date=today, user__site__in=assigned_sites).count()
        
    return render(request, "dashboard.html", context)


@login_required(login_url="admin-login")
def admin_user_face_view(request):
    if not request.user.is_staff:
        return redirect("admin-login")
    
    is_superuser = request.user.is_superuser
    site_admin_profile = None
    if not is_superuser:
        try:
            site_admin_profile = AdminProfile.objects.get(user=request.user)
        except AdminProfile.DoesNotExist:
            pass

    # Get filters
    site_filter = request.GET.get('site', 'all')
    status_filter = request.GET.get('status', 'all') # 'enrolled', 'not_enrolled', 'all'
    search_query = request.GET.get('search', '')

    employees = Employee.objects.select_related("site").order_by('name')

    # Status filtering (face enrollment)
    if status_filter == 'enrolled':
        employees = employees.filter(face_embedding__isnull=False)
    elif status_filter == 'not_enrolled':
        employees = employees.filter(face_embedding__isnull=True)

    # Site filtering
    if not is_superuser and site_admin_profile:
        assigned_sites = site_admin_profile.sites.all()
        if site_filter != 'all':
            try:
                target_id = int(site_filter)
                if not assigned_sites.filter(id=target_id).exists():
                    employees = employees.none()
                else:
                    employees = employees.filter(site_id=target_id)
            except (ValueError, TypeError):
                pass
        else:
            employees = employees.filter(site__in=assigned_sites)
    elif site_filter != 'all':
        try:
            employees = employees.filter(site_id=int(site_filter))
        except (ValueError, TypeError):
            pass

    # Search Filtering
    if search_query:
        employees = employees.filter(
            Q(name__icontains=search_query) | 
            Q(badge_number__icontains=search_query)
        )

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

    all_sites = Site.objects.all()
    
    context = {
        "employees": page_obj,
        "all_sites": all_sites,
        "selected_site": site_filter,
        "status_filter": status_filter,
        "search_query": search_query,
        "is_superuser": is_superuser,
        "site_admin_sites": site_admin_profile.sites.all() if site_admin_profile else None,
        "per_page": per_page,
    }
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        employee_data = []
        for emp in page_obj:
            employee_data.append({
                'id': emp.id,
                'name': emp.name,
                'badge_number': emp.badge_number or '-',
                'position': emp.position or '-',
                'site_name': emp.site.name if emp.site else '-',
                'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                'face_enrolled': bool(emp.face_embedding),
                'detail_url': reverse('admin-user-detail', args=[emp.id])
            })
        
        return JsonResponse({
            'employees': employee_data,
            'pagination': {
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'current_page': page_obj.number,
                'total_pages': paginator.num_pages,
                'total_items': paginator.count,
                'start_index': page_obj.start_index(),
                'end_index': page_obj.end_index(),
            },
            'filters': {
                'site': site_filter,
                'status': status_filter,
                'search': search_query,
                'per_page': per_page
            },
            'sites': list(Site.objects.values('id', 'name')) if is_superuser else [],
            'permissions': {
                'is_superuser': is_superuser
            },
            'total_count': paginator.count
        })

    # Initial Page Load (Skeleton)
    return render(request, "user_face.html", {
        "is_superuser": is_superuser,
    })


@login_required(login_url="admin-login")
def admin_user_detail_view(request, user_id):
    if not request.user.is_staff:
        # If AJAX, return 403 JSON, else redirect
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'error': 'Unauthorized'}, status=403)
        return redirect("admin-login")
    
    # We check permission but don't error out on GET for redirect simplicity,
    # but for JSON we must be strict.
    if not request.user.is_superuser:
        try:
            profile = AdminProfile.objects.get(user=request.user)
            permission_sites = profile.sites.all() # Renamed for clarity
        except AdminProfile.DoesNotExist:
            permission_sites = Site.objects.none()
    else:
        permission_sites = Site.objects.all()

    # Handle AJAX Request for Data
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        try:
            employee = Employee.objects.get(id=user_id)
            
            # Site Admin Permission Check
            if not request.user.is_superuser:
                if employee.site and not permission_sites.filter(id=employee.site.id).exists(): # Check if employee.site exists before filtering
                    return JsonResponse({'error': 'Permission Denied'}, status=403)

            filter_type = request.GET.get("filter", "daily")
            today = timezone.localdate()
            
            # Date Parsing
            date_str = request.GET.get("date")
            current_date = today
            if date_str:
                try:
                    current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    pass
            
            start_date = current_date
            end_date = current_date
            
            # Calculate Range
            if filter_type == "weekly":
                start_date = current_date - timedelta(days=current_date.weekday())
                end_date = start_date + timedelta(days=6)
            elif filter_type == "monthly":
                start_date = current_date.replace(day=1)
                # Last day of month
                next_month = start_date.replace(day=28) + timedelta(days=4)
                end_date = next_month - timedelta(days=next_month.day)
            elif filter_type == "custom":
                start_str = request.GET.get("start_date")
                end_str = request.GET.get("end_date")
                if start_str and end_str:
                    try:
                        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                    except ValueError:
                        pass

            # Fetch Records
            attendance_records = Attendance.objects.filter(
                user=employee,
                date__range=[start_date, end_date]
            ).order_by('date')
            
            # Calculate Stats
            total_records = len(attendance_records)
            present_count = sum(1 for r in attendance_records if r.status == 'present')
            late_count = sum(1 for r in attendance_records if r.status == 'late')
            absent_count = sum(1 for r in attendance_records if r.status == 'absent')
            total_late_minutes = sum(r.late_minutes for r in attendance_records)
            total_early_minutes = sum(r.early_minutes for r in attendance_records)

            # Build Response Data
            data = {
                'employee': {
                    'id': employee.id,
                    'name': employee.name,
                    'email': employee.email,
                    'phone': employee.phone,
                    'badge_number': employee.badge_number,
                    'department': employee.department,
                    'position': employee.position,
                    'site': employee.site.name if employee.site else None,
                    'site_id': employee.site.id if employee.site else None,
                    'status': employee.status,
                    'resumption_date': str(employee.resumption_date) if employee.resumption_date else None,
                    'profile_picture': employee.profile_picture.url if employee.profile_picture else None,
                    # Expanded Fields
                    'job_description': employee.job_description,
                    'salary_grade': employee.salary_grade,
                    'mol_id': employee.mol_id,
                    'labor_card_number': employee.labor_card_number,
                    'employer': employee.employer,
                    'nationality': employee.nationality,
                    'gender': employee.gender,
                    'marital_status': employee.marital_status,
                    'religion': employee.religion,
                    'date_of_birth': str(employee.date_of_birth) if employee.date_of_birth else None,
                    'date_of_joining': str(employee.date_of_joining) if employee.date_of_joining else None,
                    'passport_number': employee.passport_number,
                    'passport_expiry': str(employee.passport_expiry) if employee.passport_expiry else None,
                    'visa_details': employee.visa_details,
                    'camp': employee.camp,
                    'transportation': employee.transportation,
                },
                'stats': {
                    'total_records': total_records,
                    'present': present_count,
                    'late': late_count,
                    'absent': absent_count,
                    'late_minutes': total_late_minutes,
                    'early_minutes': total_early_minutes,
                },
                'filter': {
                    'type': filter_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'current_date': current_date,
                    'today': today
                }
            }

            if filter_type == 'daily':
                # Detailed Daily Slots
                record = attendance_records.first() if attendance_records else None
                slots = {}
                # Office In
                slots["Office In"] = {
                    "time_range": "9:00 AM",
                    "status": "present" if (record and record.check_in_time) else None,
                    "check_in": timezone.localtime(record.check_in_time).strftime("%I:%M %p") if (record and record.check_in_time) else None,
                    "late_minutes": record.late_minutes if record else 0,
                    "latitude": record.latitude if record else None,
                    "longitude": record.longitude if record else None,
                }
                if request.user.is_superuser:
                    data['employee']['gross_salary'] = str(employee.gross_salary) if employee.gross_salary else None
                    if 'salary_grade' in data['employee']: # This is already there, but just to be sure
                         pass 
                else:
                    # Explicitly remove sensitive fields if they were somehow included
                    data['employee'].pop('gross_salary', None)
                    data['employee'].pop('salary_grade', None) # Site admin shouldn't see category/grade either as per instruction "donot show salary"
                
                # Office Out
                slots["Office Out"] = {
                    "time_range": "6:00 PM",
                    "status": "present" if (record and record.check_out_time) else None,
                    "check_in": timezone.localtime(record.check_out_time).strftime("%I:%M %p") if (record and record.check_out_time) else None,
                    "early_minutes": record.early_minutes if record else 0,
                    "latitude": record.latitude if record else None,
                    "longitude": record.longitude if record else None,
                }
                data['slots'] = slots
            else:
                # Calendar/List View Data
                # Map records by date string
                records_by_date = {r.date.isoformat(): r for r in attendance_records}
                
                # Generate calendar grid if needed, or just list
                # For simplicity, we return the list of days in the range
                calendar_days = []
                curr = start_date
                while curr <= end_date:
                    record = records_by_date.get(curr.isoformat())
                    day_data = {
                        'date': curr,
                        'record': {
                            'status': record.status if record else None,
                            'late_minutes': record.late_minutes if record else 0,
                            'early_minutes': record.early_minutes if record else 0,
                        } if record else None,
                        'slots': []
                    }
                    
                    if record:
                        if record.check_in_time:
                            day_data['slots'].append({
                                'name': 'IN',
                                'time': timezone.localtime(record.check_in_time).strftime("%H:%M"),
                                'status': 'present' if record.late_minutes == 0 else 'late'
                            })
                        if record.check_out_time:
                            day_data['slots'].append({
                                'name': 'OUT',
                                'time': timezone.localtime(record.check_out_time).strftime("%H:%M"),
                                'status': 'present' if record.early_minutes == 0 else 'absent' # Logic simplification
                            })
                    
                    calendar_days.append(day_data)
                    curr += timedelta(days=1)
                    
                data['calendar_days'] = calendar_days

            return JsonResponse(data)
            
        except Employee.DoesNotExist:
            return JsonResponse({'error': 'Employee not found'}, status=404)

    # Normal GET - Return Skeleton Page
    # Pass user_id mainly for the initial JS fetch URL construction if needed, 
    # but we can also extract it from URL path in JS.
    return render(request, "user_detail.html", {'user_id': user_id})




@login_required(login_url="admin-login")
def admin_logout_view(request):
    logout(request)
    return redirect("admin-login")


# ------------------ Sites Management ------------------

@login_required(login_url="admin-login")
def admin_sites_view(request):
    """List all sites with employee counts via AJAX/Skeleton"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    is_superuser = request.user.is_superuser
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        sites_qs = Site.objects.annotate(employee_count=Count('employee')).order_by('name')
        
        # Search Filtering
        search_query = request.GET.get('search', '').strip()
        if search_query:
            sites_qs = sites_qs.filter(name__icontains=search_query)
            
        # Pagination
        per_page = int(request.GET.get('per_page', 10))
        page_num = request.GET.get('page', 1)
            
        paginator = Paginator(sites_qs, per_page)
        try:
            page_obj = paginator.page(page_num)
        except (PageNotAnInteger, EmptyPage):
            page_obj = paginator.page(1)
            
        sites_list = []
        for site in page_obj:
            sites_list.append({
                'id': site.id,
                'name': site.name,
                'employee_count': site.employee_count,
                'detail_url': reverse('admin-site-detail', args=[site.id]),
                'office_start': site.office_start_time.strftime("%H:%M") if site.office_start_time else "",
                'office_end': site.office_end_time.strftime("%H:%M") if site.office_end_time else "",
                'worker_start': site.worker_start_time.strftime("%H:%M") if site.worker_start_time else "",
                'worker_end': site.worker_end_time.strftime("%H:%M") if site.worker_end_time else "",
                'office_day_off': site.office_day_off or "",
                'worker_day_off': site.worker_day_off or ""
            })
            
        return JsonResponse({
            'results': sites_list,
            'pagination': {
                'current_page': page_obj.number,
                'num_pages': paginator.num_pages,
                'total_items': paginator.count,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'start_index': page_obj.start_index(),
                'end_index': page_obj.end_index(),
            },
            'permissions': {
                'is_superuser': is_superuser
            },
            'search_query': search_query
        })

    # Initial Page Load (Skeleton)
    return render(request, "sites.html", {
        "is_superuser": is_superuser
    })

class ImportSitesView(APIView):
    permission_classes = [IsAdminUser]

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


class ImportSitesScheduleView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=400)

        file = request.FILES['file']
        try:
            import pandas as pd
            from datetime import datetime, time
            df = pd.read_excel(file, header=None, engine='openpyxl')
            
            header_row_index = -1
            for i in range(10):
                row_values = df.iloc[i].astype(str).str.strip().tolist()
                if any('Project' in val for val in row_values) and any('Day OFF' in val for val in row_values):
                    header_row_index = i
                    break
            
            if header_row_index == -1:
                return Response({'error': 'Could not find header row with "Project" and "Day OFF"'}, status=400)
            
            data_start_index = header_row_index + 2
            header_row = df.iloc[header_row_index].astype(str).str.strip().tolist()
            
            col_map = {}
            for idx, val in enumerate(header_row):
                if 'Project' in val: col_map['project'] = idx
                if 'Office Day Off' in val: col_map['office_day_off'] = idx
                if 'Worker Day Off' in val: col_map['worker_day_off'] = idx
                if 'Office' in val:
                    col_map['office_start'] = idx
                    col_map['office_end'] = idx + 1
                if 'Site' in val:
                    col_map['worker_start'] = idx
                    col_map['worker_end'] = idx + 1

            imported_count = 0
            errors = []
            
            for index, row in df.iloc[data_start_index:].iterrows():
                try:
                    site_name = str(row.iloc[col_map['project']]).strip() if 'project' in col_map else None
                    if not site_name or site_name.lower() == 'nan': continue
                    
                    site_obj = Site.objects.filter(name__iexact=site_name).first()
                    if not site_obj:
                        site_obj = Site.objects.create(name=site_name)
                    
                    if 'office_day_off' in col_map:
                        val = row.iloc[col_map['office_day_off']]
                        site_obj.office_day_off = str(val).strip() if pd.notna(val) else site_obj.office_day_off
                    
                    if 'worker_day_off' in col_map:
                        val = row.iloc[col_map['worker_day_off']]
                        site_obj.worker_day_off = str(val).strip() if pd.notna(val) else site_obj.worker_day_off
                    
                    def parse_time(val):
                        if pd.isna(val) or str(val).lower() == 'nan': return None
                        try:
                            if isinstance(val, (datetime, time)): return val if isinstance(val, time) else val.time()
                            time_str = str(val).strip()
                            for fmt in ["%I:%M %p", "%H:%M:%S", "%H:%M"]:
                                try: return datetime.strptime(time_str, fmt).time()
                                except: pass
                            return pd.to_datetime(time_str).time()
                        except: return None

                    if 'office_start' in col_map: site_obj.office_start_time = parse_time(row.iloc[col_map['office_start']]) or site_obj.office_start_time
                    if 'office_end' in col_map: site_obj.office_end_time = parse_time(row.iloc[col_map['office_end']]) or site_obj.office_end_time
                    if 'worker_start' in col_map: site_obj.worker_start_time = parse_time(row.iloc[col_map['worker_start']]) or site_obj.worker_start_time
                    if 'worker_end' in col_map: site_obj.worker_end_time = parse_time(row.iloc[col_map['worker_end']]) or site_obj.worker_end_time
                    
                    site_obj.save()
                    imported_count += 1
                except Exception as e:
                    errors.append(f"Row {index + 1}: {str(e)}")

            return Response({'success': True, 'imported_count': imported_count, 'errors': errors})
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class DownloadSiteScheduleTemplateView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        import pandas as pd
        import io
        from django.http import HttpResponse

        header_row_1 = ["S.N.", "Project", "Office Day Off", "Worker Day Off", "Office", "", "Site", ""]
        header_row_2 = ["", "", "", "", "Duty Start", "Duty End", "Duty Start", "Duty End"]
        sample_data = [
            [1, "Elora & Velora", "Friday", "Friday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [2, "City Walk", "Friday", "Friday", "07:30 AM", "05:30 PM", "06:30 AM", "05:30 PM"],
            [3, "Alana", "Friday", "Friday", "06:30 AM", "05:00 PM", "", ""],
            [4, "Park Horizon", "Friday", "Friday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [5, "Video", "Friday", "Friday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [6, "Precast", "Sunday", "Sunday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [7, "Abu Dhabi", "Sunday", "Sunday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [8, "Factory Sauce", "Sunday", "Sunday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [9, "Opal Garden", "Sunday", "Sunday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
            [10, "The Residence", "Sunday", "Sunday", "07:00 AM", "05:00 PM", "06:30 AM", "05:00 PM"],
        ]

        df = pd.DataFrame([header_row_1, header_row_2] + sample_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, header=False, sheet_name='Site Schedules')
            workbook = writer.book
            worksheet = writer.sheets['Site Schedules']
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            worksheet.merge_range('E1:F1', 'Office', header_format)
            worksheet.merge_range('G1:H1', 'Site', header_format)
            for c, v in enumerate(header_row_1):
                if c not in [4, 5, 6, 7]: worksheet.write(0, c, v, header_format)
            for c, v in enumerate(header_row_2): worksheet.write(1, c, v, header_format)

        output.seek(0)
        response = HttpResponse(output.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=site_schedule_template.xlsx'
        return response


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
                # Re-fetch sites with pagination for error display
                sites = Site.objects.annotate(employee_count=Count('employee')).order_by('name')
                paginator = Paginator(sites, 10) # Default per_page for error
                page_obj = paginator.page(1)
                return render(request, "sites.html", {
                    'error': f'Site "{name}" already exists.',
                    'sites': page_obj,
                    'paginator': paginator,
                    'per_page': 10
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

            office_start = request.POST.get("office_start")
            office_end = request.POST.get("office_end")
            worker_start = request.POST.get("worker_start")
            worker_end = request.POST.get("worker_end")
            office_day_off = request.POST.get("office_day_off")
            worker_day_off = request.POST.get("worker_day_off")

            Site.objects.create(
                name=name, 
                coordinates=coordinates,
                office_start_time=office_start or "09:00:00",
                office_end_time=office_end or "18:00:00",
                worker_start_time=worker_start or "08:00:00",
                worker_end_time=worker_end or "17:00:00",
                office_day_off=office_day_off or "Sunday",
                worker_day_off=worker_day_off or "Sunday"
            )
            return redirect("admin-sites")
        else:
            # Re-fetch sites with pagination for error display
            sites = Site.objects.annotate(employee_count=Count('employee')).order_by('name')
            paginator = Paginator(sites, 10) # Default per_page for error
            page_obj = paginator.page(1)
            return render(request, "sites.html", {
                'error': 'Site name is required.',
                'sites': page_obj,
                'paginator': paginator,
                'per_page': 10
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
                # Re-fetch sites with pagination for error display
                sites = Site.objects.annotate(employee_count=Count('employee')).order_by('name')
                paginator = Paginator(sites, 10) # Default per_page for error
                page_obj = paginator.page(1)
                return render(request, "sites.html", {
                    'error': f'Site "{name}" already exists.',
                    'sites': page_obj,
                    'paginator': paginator,
                    'per_page': 10
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
            
            office_start = request.POST.get("office_start")
            office_end = request.POST.get("office_end")
            worker_start = request.POST.get("worker_start")
            worker_end = request.POST.get("worker_end")
            office_day_off = request.POST.get("office_day_off")
            worker_day_off = request.POST.get("worker_day_off")
            
            if office_start: site.office_start_time = office_start
            if office_end: site.office_end_time = office_end
            if worker_start: site.worker_start_time = worker_start
            if worker_end: site.worker_end_time = worker_end
            if office_day_off: site.office_day_off = office_day_off
            if worker_day_off: site.worker_day_off = worker_day_off
            
            site.save()
            return redirect("admin-sites")
        else:
            # Re-fetch sites with pagination for error display
            sites = Site.objects.annotate(employee_count=Count('employee')).order_by('name')
            paginator = Paginator(sites, 10) # Default per_page for error
            page_obj = paginator.page(1)
            return render(request, "sites.html", {
                'error': 'Site name is required.',
                'sites': page_obj,
                'paginator': paginator,
                'per_page': 10
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
    """View site details and map via AJAX/Skeleton"""
    if not request.user.is_staff:
        return redirect("admin-login")
    
    site = get_object_or_404(Site, id=site_id)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        employee_count = Employee.objects.filter(site=site).count()
        return JsonResponse({
            'site': {
                'id': site.id,
                'name': site.name,
                'employee_count': employee_count,
            }
        })

    # Initial Page Load (Skeleton)
    return render(request, "site_detail.html", {
        'site_id': site_id  # Pass ID for initial JS fetch
    })


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

# ------------------ Site Admin Management ------------------

@login_required(login_url="admin-login")
def admin_site_admins_view(request):
    """List all site admins via AJAX/Skeleton"""
    if not request.user.is_superuser:
        return redirect("admin-dashboard")
        
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        admins_qs = AdminProfile.objects.select_related('user').prefetch_related('sites').all()
        
        # Search Filtering
        search_query = request.GET.get('search', '').strip()
        if search_query:
            admins_qs = admins_qs.filter(
                Q(user__username__icontains=search_query) |
                Q(user__email__icontains=search_query) |
                Q(sites__name__icontains=search_query)
            )
            
        admins_list = []
        for admin in admins_qs:
            assigned_sites = admin.sites.all()
            first_site = assigned_sites.first()
            admins_list.append({
                'id': admin.user.id,
                'username': admin.user.username,
                'email': admin.user.email,
                'site_id': first_site.id if first_site else '', # Kept for basic compat
                'site_ids': list(assigned_sites.values_list('id', flat=True)),
                'site_name': ", ".join([s.name for s in assigned_sites]) if assigned_sites.exists() else 'No Site'
            })
            
        return JsonResponse({
            'results': admins_list,
            'sites': list(Site.objects.values('id', 'name')),
            'search_query': search_query
        })

    # Initial Skeleton
    return render(request, "site_admins.html", {
        "is_superuser": True
    })

@login_required(login_url="admin-login")
@require_http_methods(["POST"])
def admin_add_site_admin(request):
    """Add a new site admin"""
    if not request.user.is_superuser:
        return redirect("admin-dashboard")
    
    username = request.POST.get("username")
    email = request.POST.get("email")
    password = request.POST.get("password")
    site_ids = request.POST.getlist("sites")
    
    if not (username and email and password and site_ids):
        return redirect("admin-site-admins")
        
    try:
        with transaction.atomic():
            if User.objects.filter(username=username).exists():
                 # Handle error
                 pass
            
            user = User.objects.create_user(username=username, email=email, password=password)
            user.is_staff = True
            user.save()
            
            sites = Site.objects.filter(id__in=site_ids)
            profile = AdminProfile.objects.create(user=user)
            profile.sites.set(sites)
            
        return redirect("admin-site-admins")
    except Exception as e:
        print(f"Error adding site admin: {e}")
        return redirect("admin-site-admins")

@login_required(login_url="admin-login")
@require_http_methods(["POST"])
def admin_edit_site_admin(request, admin_id):
    """Edit a site admin"""
    if not request.user.is_superuser:
        return redirect("admin-dashboard")
    
    user = get_object_or_404(User, id=admin_id)
    
    username = request.POST.get("username")
    email = request.POST.get("email")
    site_ids = request.POST.getlist("sites")
    password = request.POST.get("password") # Optional
    
    try:
        with transaction.atomic():
            user.username = username
            user.email = email
            if password:
                user.set_password(password)
            user.save()
            
            # Update or create profile
            sites = Site.objects.filter(id__in=site_ids)
            profile, created = AdminProfile.objects.get_or_create(user=user)
            profile.sites.set(sites)
            profile.save()
            
        return redirect("admin-site-admins")
    except Exception as e:
        print(f"Error editing site admin: {e}")
        return redirect("admin-site-admins")

@login_required(login_url="admin-login")
@require_http_methods(["POST"])
def admin_delete_site_admin(request, admin_id):
    """Delete a site admin"""
    if not request.user.is_superuser:
        return redirect("admin-dashboard")
    
    user = get_object_or_404(User, id=admin_id)
    if not user.is_superuser: # Prevent deleting superuser
        user.delete()
    
    return redirect("admin-site-admins")

@login_required(login_url="admin-login")
@require_http_methods(["POST"])
def admin_bulk_delete_site_admins(request):
    """Bulk delete site admins"""
    if not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
        
    try:
        import json
        data = json.loads(request.body)
        ids = data.get('ids', [])
        
        if not ids:
             return JsonResponse({'error': 'No IDs provided'}, status=400)
             
        # Filter to ensure we don't delete superusers
        User.objects.filter(id__in=ids, is_superuser=False, is_staff=True).delete()
        
        return JsonResponse({'message': 'Site admins deleted successfully'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# ------------------ Excel Export ------------------

class ExportAttendanceView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        import openpyxl
        from openpyxl.utils import get_column_letter
        from django.http import HttpResponse

        user_id = request.GET.get('user_id')
        start_str = request.GET.get('start_date')
        end_str = request.GET.get('end_date')

        if not user_id:
            return Response({'error': 'User ID required'}, status=400)

        employee = get_object_or_404(Employee, id=user_id)
        
        # Check permission for site admin
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                if profile.sites.exists() and employee.site not in profile.sites.all():
                     return Response({'error': 'Unauthorized'}, status=403)
            except AdminProfile.DoesNotExist:
                pass

        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else None
        except ValueError:
            return Response({'error': 'Invalid date format'}, status=400)

        queryset = Attendance.objects.filter(user=employee).order_by('date')
        if start_date and end_date:
            queryset = queryset.filter(date__range=[start_date, end_date])

        # Create Workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"Attendance - {employee.name}"

        # Information Header
        ws.append(['Employee Name:', employee.name])
        ws.append(['Badge ID:', employee.badge_number or '-'])
        ws.append(['Housing Camp:', employee.camp or '-'])
        ws.append(['Transportation:', employee.transportation or '-'])
        ws.append([]) # Empty row

        # Headers
        headers = ['Date', 'Status', 'Check In', 'Check Out', 'Late (min)', 'Early (min)', 'Location']
        ws.append(headers)

        # Data
        for record in queryset:
            check_in = timezone.localtime(record.check_in_time).strftime("%I:%M %p") if record.check_in_time else "-"
            check_out = timezone.localtime(record.check_out_time).strftime("%I:%M %p") if record.check_out_time else "-"
            location = f"{record.latitude}, {record.longitude}" if record.latitude else "-"
            
            ws.append([
                record.date,
                record.status,
                check_in,
                check_out,
                record.late_minutes,
                record.early_minutes,
                location
            ])

        # Adjust column widths
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 15

        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=attendance_{employee.name}_{start_date}_{end_date}.xlsx'
        
        wb.save(response)
        return response

class ExportEmployeesView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        import openpyxl
        from openpyxl.utils import get_column_letter
        from django.http import HttpResponse

        # Reuse filtering logic from EmployeeListView
        employees = Employee.objects.select_related('site').all().order_by('name')
        
        # Site Filter
        site_id = request.GET.get('site')
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id and site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                         employees = Employee.objects.none()
                    else:
                        employees = employees.filter(site_id=site_id)
                else:
                    employees = employees.filter(site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                employees = Employee.objects.none()
        elif site_id and site_id != 'all':
            employees = employees.filter(site_id=site_id)

        # Status Filter
        status_filter = request.GET.get('status')
        if status_filter and status_filter != 'all':
            employees = employees.filter(status__iexact=status_filter)

        # Category Filter
        category_filter = request.GET.get('category')
        if category_filter and category_filter != 'all':
            employees = employees.filter(
                Q(salary_grade__iexact=category_filter) | 
                (Q(salary_grade__in=['', None]) & Q(category__iexact=category_filter))
            )

        # Search
        search = request.GET.get('search')
        if search:
            employees = employees.filter(
                Q(name__icontains=search) | 
                Q(email__icontains=search) | 
                Q(badge_number__icontains=search)
            )

        # Attendance Filter
        attendance_filter = request.GET.get('attendance_filter')
        if attendance_filter in ['present', 'late']:
            today = timezone.localdate()
            attendance_qs = Attendance.objects.filter(date=today)
            if attendance_filter == 'late':
                attendance_qs = attendance_qs.filter(status='late')
            else:
                attendance_qs = attendance_qs.filter(status='present')
            
            employee_ids = attendance_qs.values_list('user_id', flat=True)
            employees = employees.filter(id__in=employee_ids)

        # Create Workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Selected Employees"

        headers = ['Name', 'Badge ID', 'Site', 'Department', 'Position', 'Grade/Category', 'Status', 'Housing Camp', 'Transportation', 'Phone', 'Email']
        ws.append(headers)

        from openpyxl.styles import Font, PatternFill
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font

        for emp in employees:
            ws.append([
                emp.name,
                emp.badge_number or '-',
                emp.site.name if emp.site else '-',
                emp.department or '-',
                emp.position or '-',
                emp.salary_grade or (emp.category.capitalize() if emp.category else '-'),
                emp.status or '-',
                emp.camp or '-',
                emp.transportation or '-',
                emp.phone or '-',
                emp.email or '-'
            ])

        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 20

        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=employees_list.xlsx'
        wb.save(response)
        return response

class ExportFaceEnrollmentView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        import openpyxl
        from openpyxl.utils import get_column_letter
        from django.http import HttpResponse

        # Reuse filtering logic from admin_user_face_view
        employees = Employee.objects.select_related("site").order_by('name')
        
        # Site Filter
        site_id = request.GET.get('site', 'all')
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                         employees = Employee.objects.none()
                    else:
                        employees = employees.filter(site_id=site_id)
                else:
                    employees = employees.filter(site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                employees = Employee.objects.none()
        elif site_id != 'all':
            try:
                employees = employees.filter(site_id=int(site_id))
            except (ValueError, TypeError):
                pass

        # Enrollment Status Filter
        status_filter = request.GET.get('status', 'all')
        if status_filter == 'enrolled':
            employees = employees.filter(face_embedding__isnull=False)
        elif status_filter == 'not_enrolled':
            employees = employees.filter(face_embedding__isnull=True)

        # Search
        search = request.GET.get('search')
        if search:
            employees = employees.filter(
                Q(name__icontains=search) | 
                Q(badge_number__icontains=search)
            )

        # Create Workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Face Enrollment Status"

        headers = ['Name', 'Badge ID', 'Site', 'Enrollment Status', 'Position', 'Department', 'Housing Camp', 'Transportation']
        ws.append(headers)

        from openpyxl.styles import Font, PatternFill
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font

        for emp in employees:
            enrollment_status = "Enrolled" if emp.face_embedding else "Not Enrolled"
            ws.append([
                emp.name,
                emp.badge_number or '-',
                emp.site.name if emp.site else '-',
                enrollment_status,
                emp.position or '-',
                emp.department or '-',
                emp.camp or '-',
                emp.transportation or '-'
            ])

        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 20

        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        filename = f"Face_Enrollment_{datetime.now().strftime('%Y%m%d')}.xlsx"
        response['Content-Disposition'] = f'attachment; filename={filename}'
        
        wb.save(response)
        return response
# ------------------ Reports Module ------------------

@login_required
def admin_reports_view(request):
    """
    Renders the Reports page skeleton.
    Data is loaded via AJAX from /api/reports/data/
    """
    # Permission Check
    is_superuser = request.user.is_superuser
    try:
        admin_profile = request.user.admin_profile
    except AdminProfile.DoesNotExist:
        admin_profile = None

    if not is_superuser and not admin_profile:
        return render(request, 'dashboard.html', {'error': 'Permission Denied'})

    return render(request, 'reports.html', {'user': request.user, 'is_superuser': is_superuser})

import csv
from django.http import HttpResponse

@login_required
def export_reports_view(request):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # Permission Check
    is_superuser = request.user.is_superuser
    is_staff = request.user.is_staff
    
    try:
        admin_profile = request.user.admin_profile
        permission_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        permission_sites = Site.objects.none()

    if not is_superuser and not is_staff and not admin_profile:
        return HttpResponse("Permission Denied", status=403)

    # Filters
    date_str = request.GET.get('date')
    status_filter = request.GET.get('status')
    site_id = request.GET.get('site')
    position_filter = request.GET.get('position')
    category_filter = request.GET.get('category')

    if date_str:
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            selected_date = timezone.localdate()
    else:
        selected_date = timezone.localdate()

    # Determine visibility for the report
    employees = Employee.objects.all()
    if is_superuser or is_staff:
        # Staff/Superuser can see everything or filter by site
        if site_id and site_id != 'all':
            employees = employees.filter(site_id=site_id)
            all_sites_for_summary = Site.objects.filter(id=site_id)
        else:
            all_sites_for_summary = Site.objects.all().order_by('name')
    elif admin_profile:
        # Site admins see only their assigned sites
        employees = employees.filter(site__in=permission_sites)
        if site_id and site_id != 'all':
            employees = employees.filter(site_id=site_id)
            all_sites_for_summary = permission_sites.filter(id=site_id)
        else:
            all_sites_for_summary = permission_sites.order_by('name')
    
    if position_filter and position_filter != 'all':
        employees = employees.filter(position__iexact=position_filter)
    
    if category_filter and category_filter != 'all':
        employees = employees.filter(
            Q(salary_grade__iexact=category_filter) | 
            (Q(salary_grade__in=['', None]) & Q(category__iexact=category_filter))
        )

    attendance_records = Attendance.objects.filter(
        date=selected_date,
        user__in=employees
    ).select_related('user', 'user__site')
    attendance_map = {att.user_id: att for att in attendance_records}

    # Data Collection
    detailed_data = []
    summary_map = {} # (site_name, position) -> {present: 0, absent: 0}
    
    # Get all unique positions for the summary matrix
    all_positions = sorted(list(set(Employee.objects.exclude(position__isnull=True).exclude(position='').values_list('position', flat=True))))
    if not all_positions:
        all_positions = ["-"]

    for emp in employees:
        att = attendance_map.get(emp.id)
        status = 'Absent'
        check_in = '-'
        check_out = '-'
        site_name = emp.site.name if emp.site else '-'
        position_name = emp.position or '-'

        if att:
            status = 'Present'
            check_in = timezone.localtime(att.check_in_time).strftime('%I:%M %p') if att.check_in_time else '-'
            check_out = timezone.localtime(att.check_out_time).strftime('%I:%M %p') if att.check_out_time else '-'

        # Apply status filter for the detailed sheet
        include_in_detailed = True
        if status_filter:
            if status_filter.lower() == 'present' and status != 'Present': include_in_detailed = False
            if status_filter.lower() == 'absent' and status != 'Absent': include_in_detailed = False

        if include_in_detailed:
            detailed_data.append([
                emp.name,
                emp.badge_number,
                emp.salary_grade,
                emp.department,
                position_name,
                site_name,
                emp.camp or '-',
                emp.transportation or '-',
                selected_date,
                status,
                check_in,
                check_out
            ])

        # Aggregate for summary (always aggregate even if filtered in detailed list?)
        # User usually wants summary of the filtered set, but they said "all 45 sites"
        # If they filtered a specific site, only that site should show.
        # But if they filtered status "Absent", the summary should reflect that?
        # Typically summary reflects the population. Let's respect status/category filters if applied.
        
        # Site/Position key
        key = (site_name, position_name)
        if key not in summary_map:
            summary_map[key] = {'present': 0, 'absent': 0}
        
        if status == 'Present':
            summary_map[key]['present'] += 1
        else:
            summary_map[key]['absent'] += 1

    # Create XLSX
    wb = openpyxl.Workbook()
    
    # --- Sheet 1: Detailed Data ---
    ws_detailed = wb.active
    ws_detailed.title = "Detailed Attendance"
    
    headers = ['Employee Name', 'Badge ID', 'Grade', 'Department', 'Position', 'Site', 'Housing Camp', 'Transportation', 'Date', 'Status', 'Check In', 'Check Out']
    ws_detailed.append(headers)
    
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    for col in range(1, len(headers) + 1):
        cell = ws_detailed.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    
    for row in detailed_data:
        ws_detailed.append(row)
    
    for i, column_cells in enumerate(ws_detailed.columns):
        ws_detailed.column_dimensions[get_column_letter(i + 1)].width = 18

    # --- Sheet 2: Summary Report (Pivot Matrix) ---
    ws_summary = wb.create_sheet("Summary Report")
    summary_header_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    
    # Get Site Names for headers
    site_names = [s.name for s in all_sites_for_summary]
    if '-' not in site_names: site_names.append('-') # Handle unassigned
    site_names = sorted(site_names)

    # Header Row 1: Sites (Merged)
    ws_summary.cell(row=1, column=1, value="Attendance summary").font = Font(bold=True, size=12)
    current_col = 2
    for site_name in site_names:
        ws_summary.merge_cells(start_row=1, start_column=current_col, end_row=1, end_column=current_col + 1)
        cell = ws_summary.cell(row=1, column=current_col, value=site_name)
        cell.fill = summary_header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        current_col += 2
    
    # Grand Total Header (Merged)
    ws_summary.merge_cells(start_row=1, start_column=current_col, end_row=1, end_column=current_col + 1)
    cell = ws_summary.cell(row=1, column=current_col, value="GRAND TOTAL")
    cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center")

    # Header Row 2: Labels
    ws_summary.cell(row=2, column=1, value="Position").font = Font(bold=True)
    ws_summary.cell(row=2, column=1).fill = header_fill
    ws_summary.cell(row=2, column=1).font = header_font
    
    current_col = 2
    sub_header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    sub_header_font = Font(bold=True)
    
    for _ in range(len(site_names) + 1): # +1 for Grand Total col
        p_cell = ws_summary.cell(row=2, column=current_col, value="Present")
        a_cell = ws_summary.cell(row=2, column=current_col + 1, value="Absent")
        for cell in [p_cell, a_cell]:
            cell.fill = sub_header_fill
            cell.font = sub_header_font
            cell.alignment = Alignment(horizontal="center")
        current_col += 2

    # Data Rows
    current_row = 3
    for pos in all_positions:
        ws_summary.cell(row=current_row, column=1, value=pos)
        
        row_present_total = 0
        row_absent_total = 0
        
        current_col = 2
        for site_name in site_names:
            counts = summary_map.get((site_name, pos), {'present': 0, 'absent': 0})
            ws_summary.cell(row=current_row, column=current_col, value=counts['present'])
            ws_summary.cell(row=current_row, column=current_col + 1, value=counts['absent'])
            
            row_present_total += counts['present']
            row_absent_total += counts['absent']
            current_col += 2
            
        # Row Totals
        ws_summary.cell(row=current_row, column=current_col, value=row_present_total).font = Font(bold=True)
        ws_summary.cell(row=current_row, column=current_col + 1, value=row_absent_total).font = Font(bold=True)
        current_row += 1

    # Final Total Row (Column-wise)
    last_row = ws_summary.max_row + 1
    ws_summary.cell(row=last_row, column=1, value="TOTAL").font = Font(bold=True)
    ws_summary.cell(row=last_row, column=1).fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for col in range(2, ws_summary.max_column + 1):
        col_sum = 0
        for r in range(3, last_row):
            val = ws_summary.cell(row=r, column=col).value
            if isinstance(val, (int, float)):
                col_sum += val
        cell = ws_summary.cell(row=last_row, column=col, value=col_sum)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    ws_summary.column_dimensions['A'].width = 30
    for col in range(2, ws_summary.max_column + 1):
        ws_summary.column_dimensions[get_column_letter(col)].width = 12

    # Output
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    filename = f"attendance_report_{selected_date}.xlsx"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    wb.save(response)
    return response


class AdminSalaryReportView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        if not request.user.is_superuser:
            return redirect("admin-dashboard")
            
        month = int(request.GET.get('month', timezone.localdate().month))
        year = int(request.GET.get('year', timezone.localdate().year))
        site_id = request.GET.get('site', 'all')
        search_query = request.GET.get('search', '').strip()
        per_page = int(request.GET.get('per_page', 20))
        page_num = request.GET.get('page', 1)
            
        employees = Employee.objects.all()
        
        # Apply Filters
        if site_id != 'all':
            employees = employees.filter(site_id=site_id)
            
        if search_query:
            employees = employees.filter(
                Q(name__icontains=search_query) | 
                Q(badge_number__icontains=search_query) |
                Q(email__icontains=search_query) |
                Q(phone__icontains=search_query)
            )
            
        # Get number of days in month
        num_days = calendar.monthrange(year, month)[1]
        
        # Calculate working days (excluding Sundays)
        working_days_count = 0
        for day in range(1, num_days + 1):
            if calendar.weekday(year, month, day) != 6:  # 6 is Sunday
                working_days_count += 1
        
        # Optimize attendance counting
        attendance_stats = Attendance.objects.filter(
            date__year=year,
            date__month=month,
            status='present',
            user__in=employees
        ).values('user_id').annotate(
            count=Count('id'),
            normal_ot=Sum('normal_ot_hours'),
            special_ot=Sum('special_ot_hours')
        )
        
        attendance_map = {item['user_id']: item for item in attendance_stats}
        
        salary_data = []
        for emp in employees:
            if not emp.gross_salary:
                continue
                
            att_data = attendance_map.get(emp.id, {'count': 0, 'normal_ot': 0, 'special_ot': 0})
            present_days = att_data['count']
            normal_ot_total = float(att_data.get('normal_ot') or 0.0)
            special_ot_total = float(att_data.get('special_ot') or 0.0)
            
            daily_rate = float(emp.gross_salary) / 30.0
            absent_days = working_days_count - present_days
            deduction = daily_rate * max(0, absent_days)
            
            # OT Calculations
            # Hourly rate based on basic salary
            basic_salary = float(emp.basic_salary or emp.gross_salary)
            hourly_base = basic_salary / num_days / 8.0
            normal_ot_pay = hourly_base * normal_ot_total * 1.25
            special_ot_pay = hourly_base * special_ot_total * 1.50
            
            net_salary = float(emp.gross_salary) - deduction + normal_ot_pay + special_ot_pay
            
            salary_data.append({
                'id': emp.id,
                'name': emp.name,
                'badge_number': emp.badge_number or '-',
                'department': emp.department or '-',
                'site': emp.site.name if emp.site else '-',
                'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                'gross_salary': str(emp.gross_salary),
                'basic_salary': str(emp.basic_salary or emp.gross_salary),
                'working_days': working_days_count,
                'present_days': present_days,
                'absent_days': max(0, absent_days),
                'normal_ot_hours': round(normal_ot_total, 2),
                'special_ot_hours': round(special_ot_total, 2),
                'normal_ot_pay': round(normal_ot_pay, 2),
                'special_ot_pay': round(special_ot_pay, 2),
                'deduction': round(deduction, 2),
                'net_salary': round(net_salary, 2),
            })
            
        # AJAX Response
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            paginator = Paginator(salary_data, per_page)
            try:
                page_obj = paginator.page(page_num)
            except (PageNotAnInteger, EmptyPage):
                page_obj = paginator.page(1)
                
            return JsonResponse({
                'results': list(page_obj),
                'summary': {
                    'month': month,
                    'year': year,
                    'month_name': calendar.month_name[month],
                    'total_employees': len(salary_data),
                },
                'pagination': {
                    'current_page': page_obj.number,
                    'num_pages': paginator.num_pages,
                    'total_items': paginator.count,
                    'has_next': page_obj.has_next(),
                    'has_previous': page_obj.has_previous(),
                    'start_index': page_obj.start_index(),
                    'end_index': page_obj.end_index(),
                },
                'sites': list(Site.objects.values('id', 'name'))
            })
        
        # Initial Skeleton
        context = {
            'months': [(i, calendar.month_name[i]) for i in range(1, 13)],
            'years': range(2024, 2031),
            'current_month': timezone.localdate().month,
            'current_year': timezone.localdate().year,
        }
        return render(request, 'salary_report.html', context)

class DownloadSalarySlipView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request, employee_id, month, year):
        if not request.user.is_superuser:
            return Response({'error': 'Unauthorized'}, status=403)
            
        employee = get_object_or_404(Employee, id=employee_id)
            
        # Recalculate for the slip
        num_days = calendar.monthrange(year, month)[1]
        working_days_count = 0
        for day in range(1, num_days + 1):
            if calendar.weekday(year, month, day) != 6:
                working_days_count += 1
                
        # OT Data Fetch
        att_data = Attendance.objects.filter(
            user=employee,
            date__year=year,
            date__month=month,
            status='present'
        ).aggregate(
            count=Count('id'),
            normal_ot=Sum('normal_ot_hours'),
            special_ot=Sum('special_ot_hours')
        )
        
        present_days = att_data['count'] or 0
        normal_ot_total = float(att_data.get('normal_ot') or 0.0)
        special_ot_total = float(att_data.get('special_ot') or 0.0)
        
        gross_val = float(employee.gross_salary or 0)
        basic_salary = float(employee.basic_salary or employee.gross_salary or 0)
        daily_rate = gross_val / 30.0
        absent_days = working_days_count - present_days
        deduction = daily_rate * max(0, absent_days)
        
        # OT Pay calculation
        hourly_base = basic_salary / num_days / 8.0
        normal_ot_pay = hourly_base * normal_ot_total * 1.25
        special_ot_pay = hourly_base * special_ot_total * 1.50
        
        net_salary = gross_val - deduction + normal_ot_pay + special_ot_pay
        
        # Generate PDF using fpdf2
        pdf = FPDF()
        pdf.add_page()
        
        # Premium Header Styling
        pdf.set_fill_color(99, 102, 241) # Indigo #6366f1
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 24)
        pdf.cell(190, 25, "SALARY SLIP", ln=True, align="C")
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(190, 5, f"{calendar.month_name[month].upper()} {year}", ln=True, align="C")
        pdf.ln(20)
        
        # Reset text color
        pdf.set_text_color(30, 41, 59) # Slate 800
        
        # Employee Info Section
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(95, 10, "EMPLOYEE INFORMATION", ln=True)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("helvetica", "", 10)
        col1 = 40
        col2 = 60
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Name:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, employee.name)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Badge ID:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, employee.badge_number or "-", ln=True)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Department:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, employee.department or "-")
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Position:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, employee.position or "-", ln=True)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Site:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, employee.site.name if employee.site else "-")
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(col1, 8, "Working Days:")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(col2, 8, str(working_days_count), ln=True)
        
        pdf.ln(10)
        
        # Earnings & Deductions Table
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(190, 10, "SALARY BREAKDOWN", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Table Header
        pdf.set_fill_color(248, 250, 252)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(140, 10, "  Description", 1, 0, "L", True)
        pdf.cell(50, 10, "Amount (AED)  ", 1, 1, "R", True)
        
        # Items
        pdf.set_font("helvetica", "", 10)
        pdf.cell(140, 10, "  Basic Salary (Gross)", 1)
        pdf.cell(50, 10, f"{gross_val:,.2f}  ", 1, 1, "R")
        
        pdf.set_text_color(239, 68, 68) # Red 500
        pdf.cell(140, 10, f"  Absence Deduction ({max(0, absent_days)} days absent)", 1)
        pdf.cell(50, 10, f"-{deduction:,.2f}  ", 1, 1, "R")
        
        # OT Additions
        pdf.set_text_color(16, 185, 129) # Emerald 500
        if normal_ot_total > 0:
            pdf.cell(140, 10, f"  Normal Overtime ({normal_ot_total:.2f} hrs @ 1.25x)", 1)
            pdf.cell(50, 10, f"+{normal_ot_pay:,.2f}  ", 1, 1, "R")
        
        if special_ot_total > 0:
            pdf.cell(140, 10, f"  Special Overtime ({special_ot_total:.2f} hrs @ 1.50x)", 1)
            pdf.cell(50, 10, f"+{special_ot_pay:,.2f}  ", 1, 1, "R")
            
        # Total
        pdf.set_text_color(16, 185, 129) # Emerald 500
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(140, 12, "  NET SALARY PAYABLE", 1, 0, "L", True)
        pdf.cell(50, 12, f"{net_salary:,.2f}  ", 1, 1, "R", True)
        
        # Footer
        pdf.ln(30)
        pdf.set_text_color(100, 116, 139) # Slate 500
        pdf.set_font("helvetica", "I", 8)
        pdf.cell(190, 5, "This document is computer-generated and verified by RocketAttendance System.", ln=True, align="C")
        pdf.cell(190, 5, f"Generated on {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        
        # Border
        pdf.set_draw_color(99, 102, 241)
        pdf.set_line_width(0.5)
        pdf.rect(5, 5, 200, 287)
        
        pdf_output = pdf.output()
        buffer = io.BytesIO(pdf_output)
        
        filename = f"Salary_Slip_{employee.name.replace(' ', '_')}_{calendar.month_name[month]}_{year}.pdf"
        return FileResponse(buffer, as_attachment=True, filename=filename)
# Monthly Report Views
from datetime import datetime, timedelta
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.db.models import Q, Count
import pandas as pd
import calendar
from .models import Employee, Attendance, Site, AdminProfile

@login_required
def monthly_report_view(request):
    """Display monthly attendance report for selected site"""
    # Permission Check
    is_superuser = request.user.is_superuser
    try:
        admin_profile = request.user.admin_profile
        site_admin_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        site_admin_sites = Site.objects.none()

    if not is_superuser and not admin_profile:
        return render(request, 'dashboard.html', {'error': 'Permission Denied'})

    # Handle AJAX Request
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Get parameters
        month = int(request.GET.get('month', datetime.now().month))
        year = int(request.GET.get('year', datetime.now().year))
        site_id = request.GET.get('site')
        search_query = request.GET.get('search', '').strip()
        page_num = request.GET.get('page', 1)
        per_page = int(request.GET.get('per_page', 20))
        
        selected_site = None
        if is_superuser:
            if site_id and site_id != 'all':
                try:
                    selected_site = Site.objects.get(id=site_id)
                except Site.DoesNotExist:
                    pass
        elif site_admin_sites.exists():
            employees = employees.filter(site__in=site_admin_sites)
        
        # Get employees for selected site
        employees = Employee.objects.all()
        if selected_site:
            employees = employees.filter(site=selected_site)
            
        # Search Filter
        if search_query:
            employees = employees.filter(
                Q(name__icontains=search_query) |
                Q(badge_number__icontains=search_query) |
                Q(email__icontains=search_query) |
                Q(phone__icontains=search_query)
            )
        
        # Calculate date range for the month
        num_days = calendar.monthrange(year, month)[1]
        start_date = datetime(year, month, 1).date()
        end_date = datetime(year, month, num_days).date()
        
        # Get all attendance records for the month
        attendance_records = Attendance.objects.filter(
            date__gte=start_date,
            date__lte=end_date,
            user__in=employees
        ).select_related('user')
        
        attendance_map = {} # user_id -> list of records
        attendance_map = {} # user_id -> list of records
        for record in attendance_records:
            if record.user_id not in attendance_map:
                attendance_map[record.user_id] = []
            attendance_map[record.user_id].append(record)
            
        employee_data = []
        total_present_all = 0
        total_late_all = 0
        
        for emp in employees:
            emp_recs = attendance_map.get(emp.id, [])
            days_present = len([r for r in emp_recs if r.check_in_time])
            days_absent = num_days - days_present
            late_count = len([r for r in emp_recs if r.late_minutes > 0])
            
            attendance_percentage = (days_present / num_days * 100) if num_days > 0 else 0
            
            employee_data.append({
                'id': emp.id,
                'name': emp.name,
                'badge_number': emp.badge_number,
                'department': emp.department,
                'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                'site': emp.site.name if emp.site else '-',
                'days_present': days_present,
                'days_absent': days_absent,
                'late_count': late_count,
                'attendance_percentage': round(attendance_percentage, 2)
            })
            
            total_present_all += days_present
            total_late_all += late_count
        
        # Calculate summary statistics
        total_employees = employees.count()
        avg_attendance = (total_present_all / (total_employees * num_days) * 100) if (total_employees * num_days) > 0 else 0
        
        # Paginate results
        paginator = Paginator(employee_data, per_page)
        try:
            page_obj = paginator.page(page_num)
        except (PageNotAnInteger, EmptyPage):
            page_obj = paginator.page(1)

        data = {
            'results': list(page_obj),
            'summary': {
                'total_days': num_days,
                'total_employees': total_employees,
                'avg_attendance': round(avg_attendance, 2),
                'total_present': total_present_all,
                'total_late': total_late_all,
                'month_name': calendar.month_name[month],
                'year': year
            },
            'pagination': {
                'current_page': page_obj.number,
                'num_pages': paginator.num_pages,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'total_items': paginator.count,
                'start_index': page_obj.start_index(),
                'end_index': page_obj.end_index(),
            },
            'sites': list(Site.objects.all().values('id', 'name')) if is_superuser else [],
            'permissions': {
                'is_superuser': is_superuser
            }
        }
        return JsonResponse(data)

    # Initial Page Load (Skeleton)
    now = datetime.now()
    months = [(i, calendar.month_name[i]) for i in range(1, 13)]
    years = list(range(now.year - 2, now.year + 1))
    sites = []
    if is_superuser:
        sites = list(Site.objects.all().values('id', 'name'))
        
    context = {
        'months': months,
        'years': years,
        'sites': sites,
        'is_superuser': is_superuser,
        'current_month': now.month,
        'current_year': now.year
    }
    return render(request, 'monthly_report.html', context)


@login_required
def export_monthly_report(request):
    """Export monthly attendance report to Excel"""
    # Permission Check
    is_superuser = request.user.is_superuser
    try:
        admin_profile = request.user.admin_profile
        site_admin_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        site_admin_sites = Site.objects.none()

    if not is_superuser and not admin_profile:
        return HttpResponse("Permission Denied", status=403)

    # Get parameters
    month = int(request.GET.get('month', datetime.now().month))
    year = int(request.GET.get('year', datetime.now().year))
    site_id = request.GET.get('site')
    
    # Get selected site
    selected_site = None
    if is_superuser and site_id and site_id != 'all':
        try:
            selected_site = Site.objects.get(id=site_id)
        except Site.DoesNotExist:
            pass
    elif site_admin_sites.exists():
        employees = employees.filter(site__in=site_admin_sites)
    
    # Get employees
    employees = Employee.objects.all()
    if selected_site:
        employees = employees.filter(site=selected_site)
    
    # Calculate date range
    num_days = calendar.monthrange(year, month)[1]
    start_date = datetime(year, month, 1).date()
    end_date = datetime(year, month, num_days).date()
    
    # Get attendance records
    attendance_records = Attendance.objects.filter(
        date__gte=start_date,
        date__lte=end_date,
        user__in=employees
    ).select_related('user')
    
    # Prepare data for Excel
    data = []
    for emp in employees:
        emp_attendance = attendance_records.filter(user=emp)
        days_present = emp_attendance.filter(check_in_time__isnull=False).count()
        days_absent = num_days - days_present
        late_count = emp_attendance.filter(is_late=True).count() if hasattr(Attendance, 'is_late') else 0
        attendance_percentage = (days_present / num_days * 100) if num_days > 0 else 0
        
        data.append({
            'Employee Name': emp.name,
            'Badge ID': emp.badge_number or '-',
            'Department': emp.department or '-',
            'Site': emp.site.name if emp.site else '-',
            'Housing Camp': emp.camp or '-',
            'Transportation': emp.transportation or '-',
            'Total Days': num_days,
            'Days Present': days_present,
            'Days Absent': days_absent,
            'Late Arrivals': late_count,
            'Attendance %': round(attendance_percentage, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel file
    output = pd.ExcelWriter(f'monthly_report_{month}_{year}.xlsx', engine='xlsxwriter')
    df.to_excel(output, sheet_name='Monthly Report', index=False)
    
    # Get workbook and worksheet
    workbook = output.book
    worksheet = output.sheets['Monthly Report']
    
    # Format header
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1
    })
    
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        worksheet.set_column(col_num, col_num, 15)
    
    output.close()
    
    # Read file and return response
    with open(f'monthly_report_{month}_{year}.xlsx', 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        site_name = selected_site.name if selected_site else 'All_Sites'
        filename = f'Monthly_Report_{site_name}_{calendar.month_name[month]}_{year}.xlsx'
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    # Clean up temp file
    import os
    os.remove(f'monthly_report_{month}_{year}.xlsx')
    
    return response

class AttendanceReportDataView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        is_superuser = request.user.is_superuser
        is_staff = request.user.is_staff
        
        try:
            admin_profile = request.user.admin_profile
            permission_sites = admin_profile.sites.all()
        except AdminProfile.DoesNotExist:
            admin_profile = None
            permission_sites = Site.objects.none()

        if not is_superuser and not is_staff and not admin_profile:
             return Response({'error': 'Permission Denied'}, status=403)

        # Filters
        date_str = request.GET.get('date')
        site_id = request.GET.get('site')
        status_filter = request.GET.get('status')
        position_filter = request.GET.get('position')
        category_filter = request.GET.get('category')
        page_num = request.GET.get('page', 1)
        per_page = int(request.GET.get('per_page', 20))

        # Date Logic
        if date_str:
            try:
                selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                selected_date = timezone.localdate()
        else:
            selected_date = timezone.localdate()

        # Build Querysets
        employees = Employee.objects.all()
        sites_list = []
        raw_positions = Employee.objects.exclude(position__isnull=True).exclude(position='').values_list('position', flat=True)
        unified_positions = {}
        for p in raw_positions:
            if not p: continue
            p_strip = p.strip()
            p_lower = p_strip.lower()
            if p_lower not in unified_positions:
                unified_positions[p_lower] = p_strip.capitalize()
        positions_list = sorted(list(unified_positions.values()))

        if is_superuser or is_staff:
            sites_list = list(Site.objects.all().order_by('name').values('id', 'name'))
            if site_id and site_id != 'all':
                employees = employees.filter(site_id=site_id)
        elif permission_sites.exists():
            sites_list = list(permission_sites.order_by('name').values('id', 'name'))
            if site_id and site_id != 'all':
                employees = employees.filter(site_id=site_id)
            else:
                employees = employees.filter(site__in=permission_sites)
        
        # Category Filter
        if category_filter and category_filter != 'all':
            employees = employees.filter(
                Q(salary_grade__iexact=category_filter) | 
                (Q(salary_grade__in=['', None]) & Q(category__iexact=category_filter))
            )
        
        # Position Filter
        if position_filter and position_filter != 'all':
            employees = employees.filter(position__iexact=position_filter)

        # Attendance Fetch
        attendance_records = Attendance.objects.filter(
            date=selected_date,
            user__in=employees
        ).select_related('user', 'user__site')

        attendance_map = {att.user_id: att for att in attendance_records}

        # Process Results
        all_results = []
        stats = {'total': 0, 'present': 0, 'absent': 0, 'late': 0}

        for emp in employees:
            att = attendance_map.get(emp.id)
            status = 'Absent'
            check_in = '-'
            check_out = '-'
            
            if att:
                status = 'Present'
                check_in = timezone.localtime(att.check_in_time).strftime('%I:%M %p') if att.check_in_time else '-'
                check_out = timezone.localtime(att.check_out_time).strftime('%I:%M %p') if att.check_out_time else '-'
                if att.late_minutes > 0: stats['late'] += 1

            stats['total'] += 1
            if status == 'Present': stats['present'] += 1
            else: stats['absent'] += 1

            # Status Filter Applied after stats calculation
            if status_filter:
                if status_filter.lower() == 'present' and status != 'Present': continue
                if status_filter.lower() == 'absent' and status != 'Absent': continue

            all_results.append({
                'id': emp.id,
                'name': emp.name,
                'email': emp.email,
                'badge_number': emp.badge_number,
                'salary_grade': emp.salary_grade,
                'department': emp.department,
                'position': emp.position,
                'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                'site': emp.site.name if emp.site else '-',
                'site_id': emp.site.id if emp.site else None,
                'status': status,
                'check_in': check_in,
                'check_out': check_out,
                'latitude': att.latitude if att else None,
                'longitude': att.longitude if att else None
            })

        # Sort: Present first
        all_results.sort(key=lambda x: x['status'] != 'Present')

        # Paginate
        paginator = Paginator(all_results, per_page)
        try:
            page_obj = paginator.page(page_num)
        except (PageNotAnInteger, EmptyPage):
            page_obj = paginator.page(1)

        return Response({
            'results': list(page_obj),
            'stats': stats,
            'sites': sites_list,
            'positions': positions_list,
            'categories': sorted(list({c.strip().capitalize(): c.strip().capitalize() for c in (list(Employee.objects.exclude(category__isnull=True).exclude(category='').values_list('category', flat=True)) + list(Employee.objects.exclude(salary_grade__isnull=True).exclude(salary_grade='').values_list('salary_grade', flat=True))) if c and c.strip()}.values())),
            'selected_site': site_id,
            'selected_date': selected_date.strftime('%Y-%m-%d'),
            'permissions': {'is_superuser': is_superuser},
            'pagination': {
                'current_page': page_obj.number,
                'num_pages': paginator.num_pages,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
            }
        })

class EmployeeAttendanceHistoryView(APIView):
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
            
            # Check permission for site admin
            if not request.user.is_superuser:
                 try:
                     profile = AdminProfile.objects.get(user=request.user)
                     if profile.sites.exists() and emp.site not in profile.sites.all():
                         return Response({'error': 'Permission Denied'}, status=403)
                 except AdminProfile.DoesNotExist:
                     pass

            attendance = Attendance.objects.filter(user=emp).order_by('-date')
            
            events = []
            for att in attendance:
                # 1. Office In Event
                if att.check_in_time:
                    events.append({
                        'id': f"{att.id}_in",
                        'user_id': str(att.user.id),
                        'check_type': 'office_in',
                        'actual_time': att.check_in_time.isoformat(),
                        'is_late': att.late_minutes > 0,
                        'is_early': False,
                        'minutes_difference': att.late_minutes,
                        'face_match_confidence': 0, # Not strictly stored
                        'created_at': att.check_in_time.isoformat() # Use actual time for sorting
                    })
                
                # 2. Office Out Event
                if att.check_out_time:
                    events.append({
                        'id': f"{att.id}_out",
                        'user_id': str(att.user.id),
                        'check_type': 'office_out',
                        'actual_time': att.check_out_time.isoformat(),
                        'is_late': False,
                        'is_early': att.early_minutes > 0,
                        'minutes_difference': att.early_minutes,
                        'face_match_confidence': 0,
                        'created_at': att.check_out_time.isoformat()
                    })

            # Sort by created_at desc (newest first)
            events.sort(key=lambda x: x['created_at'], reverse=True)
            
            return Response(events)
            
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
