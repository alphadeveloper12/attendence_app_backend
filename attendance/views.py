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
from .models import Employee, Attendance, Site, FaceTemplate, AdminProfile, AppBuild, EmployeeStatusHistory, EmployeeSiteHistory
from .serializers import *
from .permissions import IsSiteAdmin
from rest_framework_simplejwt.tokens import RefreshToken
from django.http import FileResponse, Http404

# ------------------ App Build API ------------------

# Sponsor and Employer pickable values — single source of truth used by
# AdminAdd/Edit views, Excel import validation, and the AttendanceStatsView
# payload that drives the dashboard filter dropdowns.
SPONSOR_CHOICES = ('Parkway', 'Katilink', 'ReadyMix', 'Mayadan', 'Jafza', 'Golden', 'Old Emp')
EMPLOYER_CHOICES = ('PIC', 'KFD', 'Kami', 'PRMC')


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

            # Legacy "Employer" column in the spreadsheet maps to our renamed "sponsor" field.
            col_map['sponsor'] = find_col_index(['Sponsor', 'Employer'], header_row_1)
            if col_map['sponsor'] == -1: col_map['sponsor'] = find_col_index(['Sponsor', 'Employer'], header_row_2)
            # New optional "Employer" column for the parent-company choice (PIC / KFD / Kami / PRMC)
            col_map['employer'] = find_col_index(['Employer Company', 'Parent Company', 'Company'], header_row_1)
            if col_map['employer'] == -1: col_map['employer'] = find_col_index(['Employer Company', 'Parent Company', 'Company'], header_row_2)

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
                        # NOTE: the spreadsheet "Employer" column now lives on the renamed `sponsor` field.
                        emp = Employee.objects.filter(name=name, sponsor=employer_name, nationality=nationality).first()
                    
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

                    # Always update sponsor (legacy "Employer" column) and visa_details
                    emp.sponsor = get_val_from_row(row, 'sponsor') or employer_name
                    # Optional new parent-company column ("Employer Company" / "Parent Company" / "Company")
                    raw_employer_choice = (get_val_from_row(row, 'employer') or '').strip()
                    if raw_employer_choice in EMPLOYER_CHOICES:
                        emp.employer = raw_employer_choice
                    raw_sponsor_choice = (get_val_from_row(row, 'sponsor') or '').strip()
                    if raw_sponsor_choice in SPONSOR_CHOICES:
                        emp.sponsor = raw_sponsor_choice
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


# ---------------------------------------------------------------------------
# Bulk edit (template + upload)
# ---------------------------------------------------------------------------

# Column headers for the bulk-edit template. First column (Badge ID) is the
# required lookup key. Empty cells in a row mean "leave existing value alone".
# Every other column maps to an Employee field of the same logical name.
BULK_EDIT_COLUMNS = [
    'Badge ID',         # REQUIRED — primary key
    'Name',
    'Email',
    'Phone',
    'Division (Department)',
    'Position',
    'Category',
    'Site Name',
    'Site Effective From',      # required (date) when Site Name is being changed — YYYY-MM-DD
    'Sponsor',          # one of SPONSOR_CHOICES
    'Employer',         # one of EMPLOYER_CHOICES
    'Status',           # Active / Leave / Resigned / Terminated / No Renewal / Absconding / Other
    'Job Description',
    'Nationality',
    'Gender',
    'Marital Status',
    'Religion',
    'Date of Birth',            # YYYY-MM-DD
    'Date of Joining',          # YYYY-MM-DD
    'Passport Number',
    'Passport Expiry',          # YYYY-MM-DD
    'Visa Details',
    'L.Card/CEC Nr',
    'MOL ID',
    'Housing Camp',
    'Transportation',
    # Status-conditional fields — fill these only when the Status column changes
    'Resumption Date',          # required when leaving Leave → Active
    'Last Working Date',        # required for any terminal status
    'Termination Reason',       # required for Resigned / Terminated
    'Leave Approval Date',      # required when going to Leave
    'Leave Start Date',         # required when going to Leave
    'Leave End Date',           # required when going to Leave
    'Leave Type',               # required when going to Leave — Annual/Emergency/Unpaid/Hajj/Umrah
    'Leave Ticket Eligible',    # Eligible / Not Eligible
    'Leave Ticket Price',
    # Optional salary fields (only applied if request user is superuser)
    'Basic Salary',
    'Gross Salary',
]


class BulkEditTemplateView(APIView):
    """Generates an .xlsx template the admin can fill in to bulk-edit employees."""
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        import io
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from django.http import HttpResponse

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Employees'

        # Headers
        ws.append(BULK_EDIT_COLUMNS)
        header_fill = PatternFill(start_color='2563EB', end_color='2563EB', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF', size=11)
        for col_idx in range(1, len(BULK_EDIT_COLUMNS) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            ws.column_dimensions[cell.column_letter].width = max(16, len(BULK_EDIT_COLUMNS[col_idx - 1]) + 2)
        ws.freeze_panes = 'A2'

        # Instruction row
        instruction = (
            "Fill Badge ID for each row. Leave any other cell blank to keep the existing value. "
            "When you change Status, also fill the conditional date(s) per Status. "
            "Dates: YYYY-MM-DD."
        )
        ws.cell(row=2, column=1).value = instruction
        ws.cell(row=2, column=1).alignment = Alignment(wrap_text=True, vertical='top')
        ws.cell(row=2, column=1).font = Font(italic=True, color='6B7280')
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=min(8, len(BULK_EDIT_COLUMNS)))
        ws.row_dimensions[2].height = 32

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        resp = HttpResponse(
            output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        resp['Content-Disposition'] = 'attachment; filename=bulk_edit_employees_template.xlsx'
        return resp


class BulkEditEmployeesView(APIView):
    """Applies an uploaded bulk-edit .xlsx, per-row update keyed by Badge ID."""
    permission_classes = [IsAdminUser | IsSiteAdmin]

    # Maps spreadsheet column → Employee model field (simple direct assignment)
    _SIMPLE_FIELD_MAP = {
        'Name': 'name',
        'Email': 'email',
        'Phone': 'phone',
        'Division (Department)': 'department',
        'Position': 'position',
        'Category': 'salary_grade',
        'Job Description': 'job_description',
        'Nationality': 'nationality',
        'Gender': 'gender',
        'Marital Status': 'marital_status',
        'Religion': 'religion',
        'Passport Number': 'passport_number',
        'Visa Details': 'visa_details',
        'L.Card/CEC Nr': 'labor_card_number',
        'MOL ID': 'mol_id',
        'Housing Camp': 'camp',
        'Transportation': 'transportation',
    }
    _DATE_FIELD_MAP = {
        'Date of Birth': 'date_of_birth',
        'Date of Joining': 'date_of_joining',
        'Passport Expiry': 'passport_expiry',
        'Resumption Date': 'resumption_date',
        'Last Working Date': 'last_working_date',
        'Leave Approval Date': 'leave_approval_date',
        'Leave Start Date': 'leave_start_date',
        'Leave End Date': 'leave_end_date',
    }
    _DECIMAL_FIELD_MAP = {
        'Basic Salary': 'basic_salary',
        'Gross Salary': 'gross_salary',
        'Leave Ticket Price': 'leave_ticket_price',
    }

    def post(self, request):
        import openpyxl
        from datetime import datetime, date

        f = request.FILES.get('file')
        if not f:
            return Response({'error': 'No file uploaded. Use form-data with key "file".'}, status=400)

        try:
            wb = openpyxl.load_workbook(f, data_only=True)
        except Exception as e:  # noqa: BLE001
            return Response({'error': f'Could not read file: {e}'}, status=400)
        ws = wb.active

        # Build header index — header row may have an instruction row below it,
        # but our generated template puts headers on row 1.
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return Response({'error': 'Empty spreadsheet.'}, status=400)
        header = [str(c).strip() if c is not None else '' for c in rows[0]]
        # Map header name → column index
        col_index = {name: i for i, name in enumerate(header) if name}

        if 'Badge ID' not in col_index:
            return Response({'error': 'Required column "Badge ID" is missing.'}, status=400)

        def get(row, name):
            i = col_index.get(name)
            if i is None or i >= len(row):
                return None
            v = row[i]
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        def coerce_date(s):
            if not s:
                return None
            if isinstance(s, datetime):
                return s.date()
            if isinstance(s, date):
                return s
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
                try:
                    return datetime.strptime(str(s), fmt).date()
                except ValueError:
                    continue
            return None

        def coerce_decimal(s):
            if not s:
                return None
            try:
                from decimal import Decimal
                return Decimal(str(s).replace(',', '').strip())
            except Exception:  # noqa: BLE001
                return None

        def coerce_bool_eligible(s):
            if not s:
                return None
            t = str(s).strip().lower()
            if t in ('true', 'eligible', '1', 'yes'):
                return True
            if t in ('false', 'not eligible', '0', 'no'):
                return False
            return None

        # Permission filter — site admins can only edit employees in their assigned sites
        allowed_site_ids = None
        if not request.user.is_superuser:
            try:
                allowed_site_ids = set(
                    request.user.admin_profile.sites.values_list('id', flat=True)
                )
            except AdminProfile.DoesNotExist:
                allowed_site_ids = set()

        # Cache for site name lookups (case-insensitive)
        site_cache = {s.name.lower(): s for s in Site.objects.all()}

        TERMINAL_STATUSES = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}
        MASTER_STATUSES = {'Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other'}

        results = {
            'total_rows': 0,
            'updated_count': 0,
            'skipped_count': 0,
            'errors': [],
        }

        # Skip the instruction row if it happens to come after the header (our
        # generated template leaves cell A2 with text — but it has no Badge ID
        # so the lookup will naturally fail; we treat it as a no-op skip).
        for row_idx, row in enumerate(rows[1:], start=2):
            if not any(c not in (None, '') for c in row):
                continue  # fully empty row
            results['total_rows'] += 1
            badge = get(row, 'Badge ID')
            if not badge:
                # First data row of our template carries instruction text in col A
                # without a real Badge ID — silently skip it instead of erroring.
                results['skipped_count'] += 1
                continue

            try:
                emp = Employee.objects.filter(badge_number=str(badge)).first()
                if not emp:
                    results['errors'].append({
                        'row': row_idx, 'badge_number': badge,
                        'error': 'Employee not found',
                    })
                    continue

                # Site-admin scope guard
                if allowed_site_ids is not None:
                    if emp.site_id is None or emp.site_id not in allowed_site_ids:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': 'Not permitted to edit this employee (outside your assigned sites)',
                        })
                        continue

                # ── Simple text fields ───────────────────────────────────────
                for col_name, field_name in self._SIMPLE_FIELD_MAP.items():
                    v = get(row, col_name)
                    if v is not None:
                        setattr(emp, field_name, v)

                # ── Date fields ──────────────────────────────────────────────
                date_overrides = {}
                for col_name, field_name in self._DATE_FIELD_MAP.items():
                    raw = get(row, col_name)
                    if raw is None:
                        continue
                    d = coerce_date(raw)
                    if d is None:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Could not parse date in column "{col_name}": "{raw}"',
                        })
                        raise ValueError('skip')
                    date_overrides[field_name] = d

                # ── Decimal fields ───────────────────────────────────────────
                # Salary fields only applied for superuser
                for col_name, field_name in self._DECIMAL_FIELD_MAP.items():
                    raw = get(row, col_name)
                    if raw is None:
                        continue
                    if field_name in ('basic_salary', 'gross_salary') and not request.user.is_superuser:
                        continue
                    dec = coerce_decimal(raw)
                    if dec is None:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Could not parse number in column "{col_name}": "{raw}"',
                        })
                        raise ValueError('skip')
                    setattr(emp, field_name, dec)

                # ── Sponsor / Employer (validated choices) ───────────────────
                sponsor = get(row, 'Sponsor')
                if sponsor is not None:
                    if sponsor not in SPONSOR_CHOICES:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Invalid Sponsor "{sponsor}". Must be one of: {", ".join(SPONSOR_CHOICES)}',
                        })
                        raise ValueError('skip')
                    emp.sponsor = sponsor

                employer = get(row, 'Employer')
                if employer is not None:
                    if employer not in EMPLOYER_CHOICES:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Invalid Employer "{employer}". Must be one of: {", ".join(EMPLOYER_CHOICES)}',
                        })
                        raise ValueError('skip')
                    emp.employer = employer

                # ── Leave ticket eligible ────────────────────────────────────
                ticket_raw = get(row, 'Leave Ticket Eligible')
                if ticket_raw is not None:
                    parsed = coerce_bool_eligible(ticket_raw)
                    if parsed is None:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Invalid "Leave Ticket Eligible" value "{ticket_raw}". Use Eligible or Not Eligible.',
                        })
                        raise ValueError('skip')
                    emp.leave_ticket_eligible = parsed

                # ── Site by name ─────────────────────────────────────────────
                site_name = get(row, 'Site Name')
                old_site = emp.site
                if site_name is not None:
                    s_obj = site_cache.get(site_name.lower())
                    if not s_obj:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Site "{site_name}" not found',
                        })
                        raise ValueError('skip')
                    emp.site = s_obj

                # ── Status + conditional fields ──────────────────────────────
                new_status = get(row, 'Status')
                if new_status is not None:
                    if new_status not in MASTER_STATUSES:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Invalid Status "{new_status}". Must be one of: {", ".join(sorted(MASTER_STATUSES))}',
                        })
                        raise ValueError('skip')

                    old_status = emp.status

                    # Build the "effective" date values for this transition
                    res_d = date_overrides.get('resumption_date')
                    lwd   = date_overrides.get('last_working_date')
                    l_app = date_overrides.get('leave_approval_date')
                    l_st  = date_overrides.get('leave_start_date')
                    l_en  = date_overrides.get('leave_end_date')
                    l_typ = get(row, 'Leave Type')
                    t_rsn = get(row, 'Termination Reason')

                    is_resuming    = (old_status == 'Leave' and new_status == 'Active')
                    is_terminating = (old_status not in TERMINAL_STATUSES and new_status in TERMINAL_STATUSES)
                    is_starting_leave = (old_status != 'Leave' and new_status == 'Leave')
                    is_resign_or_term = (old_status != new_status and new_status in ('Resigned', 'Terminated'))

                    missing = []
                    if is_resuming and not res_d:
                        missing.append('Resumption Date')
                    if is_terminating and not lwd:
                        missing.append('Last Working Date')
                    if is_resign_or_term and not t_rsn:
                        missing.append('Termination Reason')
                    if is_starting_leave:
                        if not l_app: missing.append('Leave Approval Date')
                        if not l_st:  missing.append('Leave Start Date')
                        if not l_en:  missing.append('Leave End Date')
                        if not l_typ: missing.append('Leave Type')
                    if missing:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Status "{new_status}" requires: {", ".join(missing)}',
                        })
                        raise ValueError('skip')

                    emp.status = new_status
                    if is_resuming:
                        emp.resumption_date = res_d
                    # Clear leave/terminal artefacts when moving out of those states
                    if new_status in TERMINAL_STATUSES:
                        emp.last_working_date = lwd
                    else:
                        emp.last_working_date = None
                    if new_status in ('Resigned', 'Terminated') and t_rsn:
                        emp.termination_reason = t_rsn
                    elif new_status not in ('Resigned', 'Terminated'):
                        emp.termination_reason = None
                    if new_status == 'Leave':
                        emp.leave_approval_date = l_app
                        emp.leave_start_date    = l_st
                        emp.leave_end_date      = l_en
                        emp.leave_type          = l_typ
                    else:
                        emp.leave_approval_date = None
                        emp.leave_start_date    = None
                        emp.leave_end_date      = None
                        emp.leave_type          = None
                        emp.leave_ticket_eligible = None
                        emp.leave_ticket_price    = None

                    # History row for the transition
                    if old_status != new_status:
                        EmployeeStatusHistory.objects.create(
                            employee=emp,
                            old_status=old_status,
                            new_status=new_status,
                            leave_approval_date=l_app if is_starting_leave else None,
                            leave_start_date=l_st   if is_starting_leave else None,
                            leave_end_date=l_en     if is_starting_leave else None,
                            resumption_date=res_d   if is_resuming      else None,
                            last_working_date=lwd   if is_terminating   else None,
                            leave_type=l_typ        if is_starting_leave else None,
                            note=(
                                f"{old_status or '—'} → {new_status} (bulk edit)"
                                + (f" — Reason: {t_rsn}"
                                   if new_status in ('Resigned', 'Terminated') and t_rsn else '')
                            ),
                            changed_by=request.user if request.user.is_authenticated else None,
                        )
                else:
                    # Status not in the row — still apply non-status date overrides
                    # (DoB / DoJ / passport expiry). Skip status-conditional ones.
                    for f_name in ('date_of_birth', 'date_of_joining', 'passport_expiry'):
                        if f_name in date_overrides:
                            setattr(emp, f_name, date_overrides[f_name])

                # ── Site change history ──────────────────────────────────────
                site_changed = (
                    (old_site and emp.site and old_site.id != emp.site_id)
                    or (old_site and not emp.site)
                    or (not old_site and emp.site)
                )
                if site_changed:
                    eff_raw = get(row, 'Site Effective From')
                    eff_from = coerce_date(eff_raw) if eff_raw else None
                    if eff_raw and not eff_from:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Could not parse "Site Effective From": "{eff_raw}"',
                        })
                        raise ValueError('skip')
                    if not eff_from:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': 'Site Effective From is required when Site Name changes',
                        })
                        raise ValueError('skip')
                    EmployeeSiteHistory.objects.create(
                        employee=emp,
                        old_site=old_site,
                        new_site=emp.site,
                        effective_from=eff_from,
                        note=(
                            f"{old_site.name if old_site else '—'} → "
                            f"{emp.site.name if emp.site else '—'} (bulk edit)"
                        ),
                        changed_by=request.user if request.user.is_authenticated else None,
                    )

                emp.save()
                results['updated_count'] += 1

            except ValueError:
                # Validation error already recorded above
                continue
            except Exception as e:  # noqa: BLE001
                results['errors'].append({
                    'row': row_idx, 'badge_number': badge,
                    'error': str(e),
                })

        results['success'] = True
        return Response(results)


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

            status              = data.get('status')
            resumption_date     = parse_date(data.get('resumption_date'))
            last_working_date   = parse_date(data.get('last_working_date'))
            leave_approval_date = parse_date(data.get('leave_approval_date'))  # Last working date before leave
            leave_start_date    = parse_date(data.get('leave_start_date'))
            leave_end_date      = parse_date(data.get('leave_end_date'))
            leave_type          = data.get('leave_type') or None
            ticket_raw          = data.get('leave_ticket_eligible')
            leave_ticket_eligible = (
                True if str(ticket_raw).lower() in ('true', 'eligible', '1', 'yes')
                else False if str(ticket_raw).lower() in ('false', 'not eligible', '0', 'no')
                else None
            )
            leave_ticket_price  = parse_decimal(data.get('leave_ticket_price')) if leave_ticket_eligible else None
            termination_reason  = (data.get('termination_reason') or '').strip() or None

            # Validate last_working_date for terminal statuses
            TERMINAL_STATUSES = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}
            if status in TERMINAL_STATUSES and not last_working_date:
                return Response(
                    {'error': f'Last working date is required when status is {status}.'},
                    status=400,
                )
            # Reason required when status is Resigned or Terminated
            if status in ('Resigned', 'Terminated') and not termination_reason:
                return Response(
                    {'error': f'Reason is required when status is {status}.'},
                    status=400,
                )

            # Validate leave dates when status is Leave
            if status == 'Leave':
                missing = [n for n, v in [
                    ('Last Working Date', leave_approval_date),
                    ('Leave Start Date', leave_start_date),
                    ('Leave End Date', leave_end_date),
                ] if not v]
                if missing:
                    return Response(
                        {'error': f"Required for Leave status: {', '.join(missing)}."},
                        status=400,
                    )
                if leave_start_date and leave_end_date and leave_end_date < leave_start_date:
                    return Response(
                        {'error': 'Leave end date cannot be before leave start date.'},
                        status=400,
                    )

            new_emp = Employee.objects.create(
                name=name,
                email=email or None,
                phone=data.get('phone'),
                department=data.get('department'),
                position=data.get('position'),
                badge_number=badge,
                salary_grade=data.get('salary_grade'),
                status=status,
                resumption_date=resumption_date,
                last_working_date=last_working_date,
                termination_reason=termination_reason,
                leave_approval_date=leave_approval_date,
                leave_start_date=leave_start_date,
                leave_end_date=leave_end_date,
                leave_type=leave_type,
                leave_ticket_eligible=leave_ticket_eligible,
                leave_ticket_price=leave_ticket_price,
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
                sponsor=data.get('sponsor'),
                employer=(data.get('employer') if data.get('employer') in EMPLOYER_CHOICES else None),
                site=site,
                gross_salary=parse_decimal(data.get('gross_salary')),
                basic_salary=parse_decimal(data.get('basic_salary')),
                category=data.get('category', 'worker'),
                camp=data.get('camp'),
                transportation=data.get('transportation')
            )

            # Record initial status as history when relevant dates are present
            if status and (resumption_date or last_working_date or leave_approval_date or leave_start_date or leave_end_date):
                EmployeeStatusHistory.objects.create(
                    employee=new_emp,
                    old_status=None,
                    new_status=status,
                    leave_approval_date=leave_approval_date,
                    leave_start_date=leave_start_date,
                    leave_end_date=leave_end_date,
                    resumption_date=resumption_date,
                    last_working_date=last_working_date,
                    leave_type=leave_type,
                    leave_ticket_eligible=leave_ticket_eligible,
                    leave_ticket_price=leave_ticket_price,
                    note=(
                        f"Initial status on employee creation — Reason: {termination_reason}"
                        if status in ('Resigned', 'Terminated') and termination_reason
                        else 'Initial status on employee creation'
                    ),
                    changed_by=request.user if request.user.is_authenticated else None,
                )

            # Record the initial site assignment so the Site History tab has a starting point
            if site is not None:
                EmployeeSiteHistory.objects.create(
                    employee=new_emp,
                    old_site=None,
                    new_site=site,
                    effective_from=parse_date(data.get('date_of_joining')) or timezone.localdate(),
                    note='Initial site assignment',
                    changed_by=request.user if request.user.is_authenticated else None,
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
                'last_working_date': str(emp.last_working_date) if emp.last_working_date else '',
                'termination_reason': emp.termination_reason or '',
                'leave_approval_date': str(emp.leave_approval_date) if emp.leave_approval_date else '',
                'leave_start_date': str(emp.leave_start_date) if emp.leave_start_date else '',
                'leave_end_date': str(emp.leave_end_date) if emp.leave_end_date else '',
                'leave_type': emp.leave_type or '',
                'leave_ticket_eligible': '' if emp.leave_ticket_eligible is None else ('Eligible' if emp.leave_ticket_eligible else 'Not Eligible'),
                'leave_ticket_price': str(emp.leave_ticket_price) if emp.leave_ticket_price is not None else '',
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
                'sponsor': emp.sponsor or '',
                'employer': emp.employer or '',
                'site': emp.site.id if emp.site else '',
                'camp': emp.camp,
                'transportation': emp.transportation,
                # Document URLs (frontend uses these for "View" buttons)
                'passport_document_url': emp.passport_document.url if emp.passport_document else '',
                'visa_document_url': emp.visa_document.url if emp.visa_document else '',
                'labour_card_document_url': emp.labour_card_document.url if emp.labour_card_document else '',
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

            # ── Status transition logic ─────────────────────────────────────────
            new_status           = data.get('status')
            new_resumption       = parse_date(data.get('resumption_date'))
            new_last_working     = parse_date(data.get('last_working_date'))
            new_leave_approval   = parse_date(data.get('leave_approval_date'))
            new_leave_start      = parse_date(data.get('leave_start_date'))
            # New leave-specific fields
            new_leave_type       = data.get('leave_type') or None
            _ticket_raw          = data.get('leave_ticket_eligible')
            new_ticket_eligible  = (
                True if str(_ticket_raw).lower() in ('true', 'eligible', '1', 'yes')
                else False if str(_ticket_raw).lower() in ('false', 'not eligible', '0', 'no')
                else None
            )
            new_ticket_price     = parse_decimal(data.get('leave_ticket_price')) if new_ticket_eligible else None
            new_leave_end        = parse_date(data.get('leave_end_date'))
            new_termination_reason = (data.get('termination_reason') or '').strip() or None
            old_status           = emp.status

            # Resumption date — required when bringing employee back from Leave to Active
            is_resuming = (old_status == 'Leave' and new_status == 'Active')
            if is_resuming and not new_resumption:
                return Response(
                    {'error': 'Resumption date is required when bringing an employee back from Leave to Active.'},
                    status=400,
                )

            # Last working date — required when transitioning INTO a terminal status
            TERMINAL_STATUSES = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}
            is_terminating = (old_status not in TERMINAL_STATUSES and new_status in TERMINAL_STATUSES)
            if is_terminating and not new_last_working:
                return Response(
                    {'error': f'Last working date is required when status changes to {new_status}.'},
                    status=400,
                )
            # Reason required when transitioning INTO Resigned or Terminated
            is_resigned_or_terminated = (
                old_status != new_status and new_status in ('Resigned', 'Terminated')
            )
            if is_resigned_or_terminated and not new_termination_reason:
                return Response(
                    {'error': f'Reason is required when status changes to {new_status}.'},
                    status=400,
                )

            # Leave dates — required when transitioning INTO Leave
            is_starting_leave = (old_status != 'Leave' and new_status == 'Leave')
            if is_starting_leave:
                missing = [n for n, v in [
                    ('Last Working Date', new_leave_approval),
                    ('Leave Start Date', new_leave_start),
                    ('Leave End Date', new_leave_end),
                ] if not v]
                if missing:
                    return Response(
                        {'error': f"Required when going on Leave: {', '.join(missing)}."},
                        status=400,
                    )
                if new_leave_start and new_leave_end and new_leave_end < new_leave_start:
                    return Response(
                        {'error': 'Leave end date cannot be before leave start date.'},
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
            # last_working_date only applies to terminal statuses; clear it when the
            # employee is moved back to a non-terminal state (e.g. Active after a
            # reverted Resignation).
            if new_status in TERMINAL_STATUSES:
                emp.last_working_date = new_last_working
            else:
                emp.last_working_date = None
            # Only update termination_reason when the new status is Resigned/Terminated;
            # clear it if the employee is moved out of those statuses.
            if new_status in ('Resigned', 'Terminated'):
                emp.termination_reason = new_termination_reason or emp.termination_reason
            else:
                emp.termination_reason = None
            # If the new status is *not* Leave, clear the leave window fields so they
            # don't pollute future displays (calendar auto-LEAVE inference, etc.).
            # The historical Leave start/end dates remain preserved in EmployeeStatusHistory.
            if new_status == 'Leave':
                emp.leave_approval_date   = new_leave_approval
                emp.leave_start_date      = new_leave_start
                emp.leave_end_date        = new_leave_end
                emp.leave_type            = new_leave_type
                emp.leave_ticket_eligible = new_ticket_eligible
                emp.leave_ticket_price    = new_ticket_price
            else:
                emp.leave_approval_date   = None
                emp.leave_start_date      = None
                emp.leave_end_date        = None
                emp.leave_type            = None
                emp.leave_ticket_eligible = None
                emp.leave_ticket_price    = None
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
            emp.sponsor = data.get('sponsor')
            new_employer = data.get('employer')
            emp.employer = new_employer if new_employer in EMPLOYER_CHOICES else None

            # Document uploads (multipart) — only overwrite when a new file is sent
            if hasattr(request, 'FILES'):
                if 'passport_document' in request.FILES:
                    emp.passport_document = request.FILES['passport_document']
                if 'visa_document' in request.FILES:
                    emp.visa_document = request.FILES['visa_document']
                if 'labour_card_document' in request.FILES:
                    emp.labour_card_document = request.FILES['labour_card_document']

            # Capture the old site BEFORE we overwrite it — needed to log a site-change row.
            old_site = emp.site
            site_id = data.get('site')
            if site_id:
                try:
                    emp.site = Site.objects.get(id=site_id)
                except Site.DoesNotExist:
                    emp.site = None
            else:
                emp.site = None

            # Site-change history (only on actual transition).
            # The frontend sends `site_effective_from` (YYYY-MM-DD) — required when site
            # actually changes — so that calendar cards prior to that date keep showing
            # the previous site, and only days from that date forward show the new one.
            site_changed = (
                (old_site and emp.site and old_site.id != emp.site.id)
                or (old_site and not emp.site)
                or (not old_site and emp.site)
            )
            if site_changed:
                eff_from_raw = data.get('site_effective_from')
                effective_from = parse_date(eff_from_raw) if eff_from_raw else timezone.localdate()
                if not effective_from:
                    return Response(
                        {'error': 'Effective From date is required when changing the site.'},
                        status=400,
                    )
                EmployeeSiteHistory.objects.create(
                    employee=emp,
                    old_site=old_site,
                    new_site=emp.site,
                    effective_from=effective_from,
                    note=(
                        f"{old_site.name if old_site else '—'} → "
                        f"{emp.site.name if emp.site else '—'}"
                    ),
                    changed_by=request.user if request.user.is_authenticated else None,
                )

            emp.date_of_birth = parse_date(data.get('date_of_birth'))
            emp.date_of_joining = parse_date(data.get('date_of_joining'))
            emp.passport_expiry = parse_date(data.get('passport_expiry'))

            if request.user.is_superuser:
                emp.gross_salary = parse_decimal(data.get('gross_salary'))
                emp.basic_salary = parse_decimal(data.get('basic_salary'))
                emp.salary_grade = data.get('salary_grade', emp.salary_grade)

            emp.save()

            # Record history if status changed
            if old_status != new_status:
                EmployeeStatusHistory.objects.create(
                    employee=emp,
                    old_status=old_status,
                    new_status=new_status,
                    leave_approval_date=new_leave_approval if is_starting_leave else None,
                    leave_start_date=new_leave_start    if is_starting_leave else None,
                    leave_end_date=new_leave_end        if is_starting_leave else None,
                    resumption_date=new_resumption       if is_resuming      else None,
                    last_working_date=new_last_working   if is_terminating   else None,
                    leave_type=new_leave_type            if is_starting_leave else None,
                    leave_ticket_eligible=new_ticket_eligible if is_starting_leave else None,
                    leave_ticket_price=new_ticket_price       if is_starting_leave else None,
                    note=(
                        f"{old_status or '—'} → {new_status}"
                        + (f" — Reason: {new_termination_reason}"
                           if new_status in ('Resigned', 'Terminated') and new_termination_reason
                           else "")
                    ),
                    changed_by=request.user if request.user.is_authenticated else None,
                )

            return Response({'success': True, 'message': 'Employee updated successfully'})
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class EmployeeStatusHistoryView(APIView):
    """Returns the chronological status history for one employee."""
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        history = emp.status_history.select_related('changed_by').all()
        results = []
        for h in history:
            # Compute total days of leave when both dates are present
            total_days = None
            if h.leave_start_date and h.leave_end_date:
                total_days = (h.leave_end_date - h.leave_start_date).days + 1

            results.append({
                'id': h.id,
                'old_status': h.old_status,
                'new_status': h.new_status,
                'leave_approval_date': str(h.leave_approval_date) if h.leave_approval_date else None,
                'leave_start_date': str(h.leave_start_date) if h.leave_start_date else None,
                'leave_end_date': str(h.leave_end_date) if h.leave_end_date else None,
                'resumption_date': str(h.resumption_date) if h.resumption_date else None,
                'last_working_date': str(h.last_working_date) if h.last_working_date else None,
                'leave_type': h.leave_type or None,
                'leave_ticket_eligible': (
                    None if h.leave_ticket_eligible is None
                    else ('Eligible' if h.leave_ticket_eligible else 'Not Eligible')
                ),
                'leave_ticket_price': str(h.leave_ticket_price) if h.leave_ticket_price is not None else None,
                'total_days': total_days,
                'note': h.note,
                'changed_at': h.changed_at.isoformat() if h.changed_at else None,
                'changed_by': h.changed_by.username if h.changed_by else None,
            })
        return Response({
            'employee_id': emp.id,
            'employee_name': emp.name,
            'current_status': emp.status,
            'history': results,
        })


class EmployeeSiteHistoryView(APIView):
    """Returns the chronological site assignment history for one employee
    as a timeline of [from_site, to_site, from_date, to_date] segments.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.select_related('site').get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        rows = list(
            emp.site_history
               .select_related('old_site', 'new_site', 'changed_by')
               .order_by('effective_from', 'changed_at')
        )

        # Build segments: each transition row starts a new assignment that lasts
        # until the next row's effective_from (or "now" for the latest one).
        segments = []
        for i, h in enumerate(rows):
            start = h.effective_from
            end = (rows[i + 1].effective_from - timedelta(days=1)
                   if i + 1 < len(rows) else None)
            segments.append({
                'id': h.id,
                'site_id': h.new_site.id if h.new_site else None,
                'site_name': h.new_site.name if h.new_site else None,
                'from_date': str(start) if start else None,
                'to_date': str(end) if end else None,
                'is_current': end is None,
                'note': h.note,
                'changed_at': h.changed_at.isoformat() if h.changed_at else None,
                'changed_by': h.changed_by.username if h.changed_by else None,
                'old_site_name': h.old_site.name if h.old_site else None,
            })

        return Response({
            'employee_id': emp.id,
            'employee_name': emp.name,
            'current_site': emp.site.name if emp.site else None,
            'current_site_id': emp.site.id if emp.site else None,
            'segments': segments,
        })


class EmployeeSickLeaveView(APIView):
    """List sick leaves for an employee, or mark a date (today or past) as sick.

    GET  /api/attendance/employees/<id>/sick-leave/  → list of sick days + count
    POST /api/attendance/employees/<id>/sick-leave/  → multipart: date, certificate, [note]
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        records = (Attendance.objects
                   .filter(user=emp, status='sick')
                   .select_related('sick_leave_marked_by')
                   .order_by('-date'))
        items = []
        for r in records:
            items.append({
                'id': r.id,
                'date': str(r.date),
                'note': r.sick_leave_note or '',
                'certificate_url': (r.medical_certificate.url
                                    if r.medical_certificate else None),
                'marked_at': r.sick_leave_marked_at.isoformat() if r.sick_leave_marked_at else None,
                'marked_by': r.sick_leave_marked_by.username if r.sick_leave_marked_by else None,
            })
        return Response({
            'employee_id': emp.id,
            'employee_name': emp.name,
            'count': len(items),
            'records': items,
        })

    def post(self, request, employee_id):
        from datetime import datetime as _dt

        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        date_str = request.data.get('date')
        certificate = request.FILES.get('certificate')
        note = request.data.get('note', '') or ''

        if not date_str:
            return Response({'error': 'date is required (YYYY-MM-DD)'}, status=400)
        if not certificate:
            return Response({'error': 'Medical certificate file is required'}, status=400)

        try:
            sick_date = _dt.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)

        if sick_date > timezone.localdate():
            return Response({'error': 'Cannot mark a future date as sick leave'}, status=400)

        # Find existing record for that date or create one
        record, created = Attendance.objects.get_or_create(
            user=emp, date=sick_date,
            defaults={'status': 'sick'},
        )

        # Only allow flipping from absent / late / sick → sick.
        # Don't overwrite a 'present' day (that would erase real attendance).
        if not created and record.status == 'present':
            return Response(
                {'error': "Cannot mark a 'present' day as sick. Adjust the attendance record first."},
                status=400,
            )

        record.status = 'sick'
        record.medical_certificate = certificate
        record.sick_leave_note = note
        record.sick_leave_marked_at = timezone.now()
        record.sick_leave_marked_by = request.user if request.user.is_authenticated else None
        record.save()

        return Response({
            'success': True,
            'id': record.id,
            'date': str(record.date),
            'certificate_url': record.medical_certificate.url if record.medical_certificate else None,
            'note': record.sick_leave_note or '',
            'marked_at': record.sick_leave_marked_at.isoformat() if record.sick_leave_marked_at else None,
            'marked_by': record.sick_leave_marked_by.username if record.sick_leave_marked_by else None,
        }, status=201)

    def delete(self, request, employee_id):
        """Remove a sick-leave entry by attendance id (revert to absent)."""
        record_id = request.query_params.get('id') or request.data.get('id')
        if not record_id:
            return Response({'error': 'id is required'}, status=400)
        try:
            record = Attendance.objects.get(id=record_id, user_id=employee_id, status='sick')
        except Attendance.DoesNotExist:
            return Response({'error': 'Sick leave record not found'}, status=404)

        if record.medical_certificate:
            record.medical_certificate.delete(save=False)
        record.medical_certificate = None
        record.sick_leave_note = None
        record.sick_leave_marked_at = None
        record.sick_leave_marked_by = None
        record.status = 'absent'
        record.save()
        return Response({'success': True})


class MarkDayView(APIView):
    """Set an attendance day's status to present / absent / leave (no file required).

    POST /api/attendance/employees/<id>/mark-day/
        body: { "date": "YYYY-MM-DD", "action": "present"|"absent"|"leave" }

    Notes:
        - 'sick' marking remains on EmployeeSickLeaveView because it requires a certificate.
        - This endpoint will NOT overwrite a real 'present' record produced by face-scan attendance:
          if the existing record has a `check_in_time`, the admin must clear it first
          (we don't want a click to nuke a verified attendance entry).
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    ALLOWED_ACTIONS = ('present', 'absent', 'leave')

    def post(self, request, employee_id):
        from datetime import datetime as _dt

        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        date_str = request.data.get('date')
        action = (request.data.get('action') or '').strip().lower()

        if not date_str:
            return Response({'error': 'date is required (YYYY-MM-DD)'}, status=400)
        if action not in self.ALLOWED_ACTIONS:
            return Response(
                {'error': f"action must be one of {', '.join(self.ALLOWED_ACTIONS)}"},
                status=400,
            )

        try:
            target_date = _dt.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return Response({'error': 'Invalid date format. Use YYYY-MM-DD'}, status=400)

        if target_date > timezone.localdate():
            return Response({'error': 'Cannot mark a future date'}, status=400)

        record, created = Attendance.objects.get_or_create(
            user=emp, date=target_date,
            defaults={'status': action},
        )

        # Refuse to clobber a real face-scan check-in.
        if (not created and record.check_in_time
                and record.status == 'present' and action != 'present'):
            return Response(
                {'error': "This day has a real check-in time. Clear the attendance first if you want to change it."},
                status=400,
            )

        # Clear sick-leave artefacts when leaving the 'sick' state
        if record.status == 'sick' and action != 'sick':
            if record.medical_certificate:
                record.medical_certificate.delete(save=False)
            record.medical_certificate = None
            record.sick_leave_note = None
            record.sick_leave_marked_at = None
            record.sick_leave_marked_by = None

        # Clear check-in/out when moving away from present/late
        if action in ('absent', 'leave'):
            record.check_in_time = None
            record.check_out_time = None
            record.late_minutes = 0
            record.early_minutes = 0

        record.status = action
        record.save()

        return Response({
            'success': True,
            'id': record.id,
            'date': str(record.date),
            'status': record.status,
        }, status=200)


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
        sponsor = data.get("sponsor") or ""
        employer_choice = data.get("employer") or ""
        if employer_choice not in EMPLOYER_CHOICES:
            employer_choice = None
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
            emp.sponsor = sponsor
            emp.employer = employer_choice
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
                sponsor=sponsor,
                employer=employer_choice,
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
            employer_filter = request.GET.get('employer')
            
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

            if employer_filter and employer_filter != 'all':
                employees = employees.filter(employer__iexact=employer_filter)
                attendance = attendance.filter(user__employer__iexact=employer_filter)

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
            MASTER_STATUSES = ['Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other']
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
                    "employers": list(EMPLOYER_CHOICES),
                    "sponsors": list(SPONSOR_CHOICES),
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
        site_id = request.GET.get('site')

        # ── Pool 1: Out-of-bounds (geofence failure) ─────────────────────────
        geofence_alerts = Attendance.objects.filter(
            date=today, is_within_geofence=False,
        ).select_related('user', 'user__site')

        # ── Pool 2: Employee currently on Leave but face-scanned today ───────
        # An employee marked an attendance while the system has them on Leave —
        # admin needs to know either to revoke leave or correct the attendance.
        on_leave_alerts = (
            Attendance.objects
            .filter(date=today, user__status='Leave')
            .filter(
                Q(check_in_time__isnull=False) | Q(check_out_time__isnull=False)
            )
            .filter(
                Q(user__leave_start_date__isnull=True)
                | Q(user__leave_start_date__lte=today)
            )
            .filter(
                Q(user__leave_end_date__isnull=True)
                | Q(user__leave_end_date__gte=today)
            )
            .select_related('user', 'user__site')
        )

        # Site-admin scope filter
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id and site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                        geofence_alerts = geofence_alerts.none()
                        on_leave_alerts = on_leave_alerts.none()
                    else:
                        geofence_alerts = geofence_alerts.filter(user__site_id=site_id)
                        on_leave_alerts = on_leave_alerts.filter(user__site_id=site_id)
                else:
                    geofence_alerts = geofence_alerts.filter(user__site__in=assigned_sites)
                    on_leave_alerts = on_leave_alerts.filter(user__site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                geofence_alerts = geofence_alerts.none()
                on_leave_alerts = on_leave_alerts.none()
        elif site_id and site_id != 'all':
            geofence_alerts = geofence_alerts.filter(user__site_id=site_id)
            on_leave_alerts = on_leave_alerts.filter(user__site_id=site_id)

        data = []
        for a in geofence_alerts:
            data.append({
                "id": a.id,
                "user_name": a.user.name,
                "user_id": a.user.id,
                "user_pic": a.user.profile_picture.url if a.user.profile_picture else None,
                "site": a.user.site.name if a.user.site else "-",
                "time": a.check_in_time.strftime("%H:%M") if a.check_in_time else (a.check_out_time.strftime("%H:%M") if a.check_out_time else "-"),
                "lat": a.latitude,
                "long": a.longitude,
                "status": "Out of Bounds",
                "kind": "geofence",
            })

        # Avoid duplicating the same Attendance row if it triggers both alerts
        geofence_ids = {a["id"] for a in data}
        for a in on_leave_alerts:
            if a.id in geofence_ids:
                continue
            data.append({
                "id": a.id,
                "user_name": a.user.name,
                "user_id": a.user.id,
                "user_pic": a.user.profile_picture.url if a.user.profile_picture else None,
                "site": a.user.site.name if a.user.site else "-",
                "time": a.check_in_time.strftime("%H:%M") if a.check_in_time else (a.check_out_time.strftime("%H:%M") if a.check_out_time else "-"),
                "lat": a.latitude,
                "long": a.longitude,
                "status": "Marked attendance while on Leave",
                "kind": "on_leave",
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

        # Filter by employer (one of EMPLOYER_CHOICES)
        employer_filter = request.GET.get('employer')
        if employer_filter and employer_filter != 'all':
            employees = employees.filter(employer__iexact=employer_filter)

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
            sick_count = sum(1 for r in attendance_records if r.status == 'sick')
            total_late_minutes = sum(r.late_minutes for r in attendance_records)
            total_early_minutes = sum(r.early_minutes for r in attendance_records)
            # Total sick leaves taken (across all time, not just filtered range)
            total_sick_count = Attendance.objects.filter(user=employee, status='sick').count()

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
                    'last_working_date': str(employee.last_working_date) if employee.last_working_date else None,
                    'termination_reason': employee.termination_reason or None,
                    'leave_approval_date': str(employee.leave_approval_date) if employee.leave_approval_date else None,
                    'leave_start_date': str(employee.leave_start_date) if employee.leave_start_date else None,
                    'leave_end_date': str(employee.leave_end_date) if employee.leave_end_date else None,
                    'leave_type': employee.leave_type or None,
                    'leave_ticket_eligible': (
                        None if employee.leave_ticket_eligible is None
                        else ('Eligible' if employee.leave_ticket_eligible else 'Not Eligible')
                    ),
                    'leave_ticket_price': str(employee.leave_ticket_price) if employee.leave_ticket_price is not None else None,
                    'leave_total_days': (
                        (employee.leave_end_date - employee.leave_start_date).days + 1
                        if employee.leave_start_date and employee.leave_end_date else None
                    ),
                    'profile_picture': employee.profile_picture.url if employee.profile_picture else None,
                    # Expanded Fields
                    'job_description': employee.job_description,
                    'salary_grade': employee.salary_grade,
                    'mol_id': employee.mol_id,
                    'labor_card_number': employee.labor_card_number,
                    'sponsor': employee.sponsor,
                    'employer': employee.employer,
                    'passport_document_url': employee.passport_document.url if employee.passport_document else None,
                    'visa_document_url': employee.visa_document.url if employee.visa_document else None,
                    'labour_card_document_url': employee.labour_card_document.url if employee.labour_card_document else None,
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
                    'sick': sick_count,
                    'total_sick': total_sick_count,
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
                # Sick-leave info for the day (if any)
                if record and record.status == 'sick':
                    data['sick_leave'] = {
                        'date': str(record.date),
                        'note': record.sick_leave_note or '',
                        'certificate_url': record.medical_certificate.url if record.medical_certificate else None,
                        'marked_at': record.sick_leave_marked_at.isoformat() if record.sick_leave_marked_at else None,
                        'marked_by': record.sick_leave_marked_by.username if record.sick_leave_marked_by else None,
                    }
            else:
                # Calendar/List View Data
                # Map records by date string
                records_by_date = {r.date.isoformat(): r for r in attendance_records}

                # Build the list of leave (start, end) intervals.
                #
                # Rules:
                #   1. History 'Leave' transitions are clamped by any subsequent
                #      transition (Active resumption, terminal status, or a new Leave).
                #      An employee who returned early shouldn't have the rest of the
                #      planned leave window still showing as LEAVE on their calendar.
                #   2. The employee's *current* leave_start_date / leave_end_date is
                #      only used if status is still 'Leave'. Stale values from previous
                #      leaves don't pollute the calendar.
                leave_intervals = []
                history_list = list(
                    EmployeeStatusHistory.objects
                    .filter(employee=employee)
                    .order_by('changed_at')
                )
                for i, h in enumerate(history_list):
                    if (h.new_status != 'Leave'
                            or not h.leave_start_date or not h.leave_end_date):
                        continue
                    effective_end = h.leave_end_date
                    for later in history_list[i + 1:]:
                        if later.new_status == 'Leave':
                            # A new leave started; clamp this one just before it.
                            if later.leave_start_date:
                                effective_end = min(
                                    effective_end,
                                    later.leave_start_date - timedelta(days=1),
                                )
                            break
                        if later.new_status == 'Active' and later.resumption_date:
                            effective_end = min(
                                effective_end,
                                later.resumption_date - timedelta(days=1),
                            )
                            break
                        if later.new_status in ('Resigned', 'Terminated', 'No Renewal', 'Absconding'):
                            if later.last_working_date:
                                effective_end = min(effective_end, later.last_working_date)
                            break
                    if effective_end >= h.leave_start_date:
                        leave_intervals.append((h.leave_start_date, effective_end))

                if (employee.status == 'Leave'
                        and employee.leave_start_date and employee.leave_end_date):
                    leave_intervals.append(
                        (employee.leave_start_date, employee.leave_end_date)
                    )

                # Per-day site resolution — picks the assignment that was effective
                # on that day from EmployeeSiteHistory. Falls back to the current
                # employee.site when the date predates any recorded history.
                site_segments = list(
                    employee.site_history
                            .select_related('new_site')
                            .order_by('effective_from')
                )
                fallback_site_name = employee.site.name if employee.site else None

                def _site_for_day(d):
                    matched = None
                    for h in site_segments:
                        if h.effective_from <= d:
                            matched = h
                        else:
                            break
                    if matched:
                        return matched.new_site.name if matched.new_site else None
                    return fallback_site_name

                # Per-day employee status resolution from EmployeeStatusHistory
                # (status changes are sequential by changed_at). For dates before any
                # recorded change we use the employee's *current* status as a best-guess.
                status_changes = list(
                    EmployeeStatusHistory.objects
                    .filter(employee=employee)
                    .order_by('changed_at')
                    .values('new_status', 'changed_at')
                )

                def _status_for_day(d):
                    eff = None
                    for ch in status_changes:
                        if ch['changed_at'].date() <= d:
                            eff = ch['new_status']
                        else:
                            break
                    return eff or employee.status

                # Trim days that fall after the employee's last working date when the
                # employee is in a terminal status — they were no longer with the
                # company on those days, so no card should render.
                TERMINAL_TRIM = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}
                terminal_cutoff = (
                    employee.last_working_date
                    if employee.status in TERMINAL_TRIM and employee.last_working_date
                    else None
                )

                # Generate calendar grid if needed, or just list
                # For simplicity, we return the list of days in the range
                calendar_days = []
                curr = start_date
                while curr <= end_date:
                    # Skip days after the employee's terminal cut-off
                    if terminal_cutoff and curr > terminal_cutoff:
                        curr += timedelta(days=1)
                        continue

                    record = records_by_date.get(curr.isoformat())

                    is_on_leave = any(s <= curr <= e for (s, e) in leave_intervals)

                    # Effective status — explicit record beats inferred leave
                    if record:
                        eff_status = record.status
                    elif is_on_leave:
                        eff_status = 'leave'
                    else:
                        eff_status = None

                    day_data = {
                        'date': curr,
                        'is_on_leave': is_on_leave,
                        'site_name': _site_for_day(curr),
                        'employee_status': _status_for_day(curr),
                        'record': {
                            'status': eff_status,
                            'late_minutes': record.late_minutes if record else 0,
                            'early_minutes': record.early_minutes if record else 0,
                            'medical_certificate_url': (
                                record.medical_certificate.url
                                if record and record.medical_certificate else None
                            ),
                            'sick_leave_note': record.sick_leave_note if record else None,
                        } if (record or is_on_leave) else None,
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
    return render(request, "user_detail.html", {
        'user_id': user_id,
        'is_superuser': request.user.is_superuser
    })




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
                'worker_day_off': site.worker_day_off or "",
                'has_geofence': bool(site.coordinates),
                'geofence_filename': site.geofence_filename or "",
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
            geofence_filename = None
            if 'kml_file' in request.FILES:
                try:
                    kml_file = request.FILES['kml_file']
                    geofence_filename = kml_file.name
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
                geofence_filename=geofence_filename,
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
                    site.geofence_filename = kml_file.name
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
    employer_filter = request.GET.get('employer')

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

    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)

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
        employer_filter = request.GET.get('employer')
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

        # Employer Filter (parent company)
        if employer_filter and employer_filter != 'all':
            employees = employees.filter(employer__iexact=employer_filter)

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
            'employers': list(EMPLOYER_CHOICES),
            'sponsors': list(SPONSOR_CHOICES),
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
    employer_filter = request.GET.get('employer')

    # Get selected site
    selected_site = None
    if is_superuser and site_id and site_id != 'all':
        try:
            selected_site = Site.objects.get(id=site_id)
        except Site.DoesNotExist:
            pass

    # Get employees
    employees = Employee.objects.all()
    if not is_superuser and site_admin_sites.exists():
        employees = employees.filter(site__in=site_admin_sites)
    if selected_site:
        employees = employees.filter(site=selected_site)
    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)
    
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
        employer_filter = request.GET.get('employer')
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

        # Employer Filter (one of EMPLOYER_CHOICES)
        if employer_filter and employer_filter != 'all':
            employees = employees.filter(employer__iexact=employer_filter)

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
            'employers': list(EMPLOYER_CHOICES),
            'sponsors': list(SPONSOR_CHOICES),
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
