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
import json
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
from .models import Employee, Attendance, Site, FaceTemplate, AdminProfile, AppBuild, EmployeeStatusHistory, EmployeeSiteHistory, EmployeeAttachment, EmployeeSalaryHistory, JobCategory, Department, AppSettings, PublicHoliday, DistributionSnapshot, EmployeeGeneralNote
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
            created_count = 0
            updated_count = 0
            errors = []

            # --- BATCH LOOKUPS + CASE-INSENSITIVE CANONICAL CACHES ---
            # Badge ID is the unique identifier. We normalize each badge to a
            # canonical form (stripped, no inner whitespace, lowercased) so that
            # "8033", " 8033", "8033 " all collapse to one record.
            data_df = df.iloc[start_data_index:]

            def _norm_badge(b):
                if not b:
                    return ''
                # Excel often loads integers as "8033.0" — strip the .0
                s = str(b).strip()
                if s.endswith('.0') and s[:-2].isdigit():
                    s = s[:-2]
                return s.replace(' ', '').lower()

            all_badge_keys = set()
            for _, row in data_df.iterrows():
                k = _norm_badge(get_val_from_row(row, 'badge_number'))
                if k:
                    all_badge_keys.add(k)

            # Build a badge lookup that's case/space-insensitive against the DB.
            existing_emps_by_badge = {}
            for e in Employee.objects.exclude(badge_number__isnull=True).exclude(badge_number=''):
                k = _norm_badge(e.badge_number)
                if k in all_badge_keys:
                    existing_emps_by_badge[k] = e

            # --- Case-insensitive canonical caches for free-text fields ---
            # Key = value.lower(), Value = the FIRST-SEEN original spelling.
            # When the sheet contains "vida" but the DB already has "Vida",
            # we re-use "Vida" instead of creating a new variant.
            site_cache = {s.name.lower(): s for s in Site.objects.all()}

            def _build_cache(qs_values):
                cache = {}
                for v in qs_values:
                    if not v: continue
                    v = str(v).strip()
                    if not v: continue
                    k = v.lower()
                    cache.setdefault(k, v)
                return cache

            import re as _re
            _SPACE_RE = _re.compile(r'\s+')

            def _norm_str(value):
                """Strip leading/trailing whitespace AND collapse internal runs
                of whitespace so "Head  Office" and "Head Office " collapse."""
                if value is None:
                    return None
                v = _SPACE_RE.sub(' ', str(value)).strip()
                return v or None

            def _build_cache_from_field(model, field):
                cache = {}
                qs = (model.objects
                      .exclude(**{f'{field}__isnull': True})
                      .exclude(**{field: ''})
                      .values_list(field, flat=True).distinct())
                for v in qs:
                    n = _norm_str(v)
                    if n:
                        cache.setdefault(n.lower(), n)
                return cache

            department_cache = _build_cache_from_field(Employee, 'department')
            for v in JobCategory.objects.exclude(department__isnull=True).exclude(department='').values_list('department', flat=True).distinct():
                n = _norm_str(v)
                if n: department_cache.setdefault(n.lower(), n)

            position_cache = _build_cache_from_field(Employee, 'position')
            for v in JobCategory.objects.values_list('name', flat=True).distinct():
                n = _norm_str(v)
                if n: position_cache.setdefault(n.lower(), n)

            sponsor_cache       = _build_cache_from_field(Employee, 'sponsor')
            employer_cache      = _build_cache_from_field(Employee, 'employer')
            nationality_cache   = _build_cache_from_field(Employee, 'nationality')
            gender_cache        = _build_cache_from_field(Employee, 'gender')
            marital_cache       = _build_cache_from_field(Employee, 'marital_status')
            religion_cache      = _build_cache_from_field(Employee, 'religion')
            status_cache        = _build_cache_from_field(Employee, 'status')
            # Seed status_cache with the canonical master list so a freshly
            # imported sheet still maps "active"/"ACTIVE" → "Active".
            for canonical in ('Active', 'Leave', 'Resigned', 'Terminated',
                              'No Renewal', 'Absconding'):
                status_cache.setdefault(canonical.lower(), canonical)
            # Same idea for categories
            category_cache = {'staff': 'staff', 'worker': 'worker'}

            def _canon(value, cache):
                """Return the canonical (first-seen) spelling for `value`. If new,
                remember it in the cache so subsequent rows of the same import
                match. Case/whitespace insensitive."""
                v = _norm_str(value)
                if v is None:
                    return None
                k = v.lower()
                if k in cache:
                    return cache[k]
                cache[k] = v
                return v

            def parse_float(val):
                if not val: return None
                try:
                    return float(str(val).replace(',', ''))
                except Exception:  # noqa: BLE001
                    return None

            def parse_date(date_str):
                if not date_str: return None
                try:
                    return pd.to_datetime(date_str).date()
                except Exception:  # noqa: BLE001
                    return None

            for index, row in data_df.iterrows():
                try:
                    name = get_val_from_row(row, 'name')
                    if not name: continue

                    badge = get_val_from_row(row, 'badge_number')
                    badge_key = _norm_badge(badge)
                    if not badge_key:
                        errors.append(f"Row {index}: skipped — Badge ID is required (it's the unique identifier).")
                        continue

                    emp = existing_emps_by_badge.get(badge_key)
                    is_new = emp is None
                    if is_new:
                        emp = Employee()
                        # Keep the badge in the form the sheet provided (stripped).
                        emp.badge_number = str(badge).strip()
                        existing_emps_by_badge[badge_key] = emp

                    # --- Always assign every field that has a value in this row.
                    # Empty cells leave the existing value alone for updates, but
                    # default to None for newly-created rows.
                    def _set(field, value):
                        if value is not None and value != '':
                            setattr(emp, field, value)
                        elif is_new:
                            setattr(emp, field, value if value != '' else None)

                    _set('name', _norm_str(name))
                    _set('department',     _canon(get_val_from_row(row, 'department'),     department_cache))
                    _set('position',       _canon(get_val_from_row(row, 'position'),       position_cache))
                    _set('salary_grade',   _canon(get_val_from_row(row, 'salary_grade'),   position_cache))
                    _set('job_description', _norm_str(get_val_from_row(row, 'job_description')))
                    _set('nationality',    _canon(get_val_from_row(row, 'nationality'),    nationality_cache))
                    _set('gender',         _canon(get_val_from_row(row, 'gender'),         gender_cache))
                    _set('marital_status', _canon(get_val_from_row(row, 'marital_status'), marital_cache))
                    _set('religion',       _canon(get_val_from_row(row, 'religion'),       religion_cache))
                    _set('labor_card_number', _norm_str(get_val_from_row(row, 'labor_card_number')))
                    _set('mol_id',         _norm_str(get_val_from_row(row, 'mol_id')))
                    _set('passport_number', _norm_str(get_val_from_row(row, 'passport_number')))
                    _set('visa_details',   _norm_str(get_val_from_row(row, 'visa_details')))

                    raw_status = _canon(get_val_from_row(row, 'status'), status_cache)
                    if raw_status:
                        emp.status = raw_status

                    # Salary components
                    gross = parse_float(get_val_from_row(row, 'gross_salary'))
                    basic = parse_float(get_val_from_row(row, 'basic_salary'))
                    if gross is not None:
                        emp.gross_salary = gross
                    if basic is not None:
                        emp.basic_salary = basic
                    elif is_new and gross is not None:
                        emp.basic_salary = gross

                    # Category — auto-derive only on first create. Subsequent
                    # imports leave the admin's manual Worker/Staff choice alone.
                    if is_new:
                        div = (emp.department or '').lower()
                        cat = (emp.salary_grade or '').lower()
                        emp.category = 'staff' if 'staff' in div or 'staff' in cat or 'office' in div else 'worker'

                    # Site — canonical match against existing Sites (case+space-insensitive).
                    # If the sheet says "alana" but the system already has "Alana",
                    # assign the existing "Alana" — do NOT create a new Site.
                    site_name = _norm_str(get_val_from_row(row, 'site'))
                    if site_name:
                        if site_name.upper() in ('HO', 'HEAD OFFICE'):
                            site_name = 'Head Office'
                        site_key = site_name.lower()
                        if site_key in site_cache:
                            emp.site = site_cache[site_key]
                        else:
                            site_obj = Site.objects.create(name=site_name)
                            site_cache[site_key] = site_obj
                            emp.site = site_obj

                    # Dates
                    dob = parse_date(get_val_from_row(row, 'dob'))
                    doj = parse_date(get_val_from_row(row, 'doj'))
                    pex = parse_date(get_val_from_row(row, 'passport_expiry'))
                    if dob: emp.date_of_birth = dob
                    if doj: emp.date_of_joining = doj
                    if pex: emp.passport_expiry = pex

                    if is_new and not emp.phone:
                        emp.phone = "0000000000"

                    # Sponsor — legacy "Employer" column from old sheets. Honour the
                    # constrained SPONSOR_CHOICES list if it matches, otherwise
                    # canonicalize free-text via the cache.
                    raw_sponsor = get_val_from_row(row, 'sponsor') or employer_name
                    if raw_sponsor:
                        s = str(raw_sponsor).strip()
                        # Case-insensitive match against the constrained choices
                        match = next((c for c in SPONSOR_CHOICES if c.lower() == s.lower()), None)
                        emp.sponsor = match or _canon(s, sponsor_cache)

                    # Employer — the new parent-company choice (PIC / KFD / Kami / PRMC).
                    raw_employer = get_val_from_row(row, 'employer')
                    if raw_employer:
                        s = str(raw_employer).strip()
                        match = next((c for c in EMPLOYER_CHOICES if c.lower() == s.lower()), None)
                        if match:
                            emp.employer = match
                        else:
                            # Fall back to whatever the sheet had, canonicalized
                            emp.employer = _canon(s, employer_cache)

                    emp.save()
                    success_count += 1
                    if is_new:
                        created_count += 1
                    else:
                        updated_count += 1

                except Exception as e:
                    errors.append(f"Row {index}: {str(e)}")
            
            return Response({
                'success': True,
                'imported_count': success_count,
                'created_count': created_count,
                'updated_count': updated_count,
                'errors': errors[:10],
            })

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
    'Passport Expiry',          # DD-MM-YYYY
    'Visa Details',
    'Visa Expiry',              # DD-MM-YYYY
    'L.Card/CEC Nr',
    'Labour Card Expiry',       # DD-MM-YYYY
    'MOL ID',
    'Emirates ID',              # 784-YYYY-NNNNNNN-C
    'Housing Camp',
    'Transportation',
    'Working Type',             # Permanent / Contract / Temporary
    'Working Shift',            # Day / Night / Rotating
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
    'Accommodation Allowance',
    'Transport Allowance',
    'Food Allowance',
    'Fixed OT Allowance',
    'Other Allowance',
    'Salary Reduction',
    'Gross Salary',
    # Insurance — WC (Workmen's Compensation)
    'WC Insurance Name',
    'WC Insurance Status',       # Active / To be added / To be cancelled / Not Applicable
    'WC Insurance Start',        # DD-MM-YYYY
    'WC Insurance End',          # DD-MM-YYYY
    'WC Premium Cost',
    # Insurance — Medical
    'Medical Insurance Name',
    'Medical Card Number',
    'Medical Insurance Status',  # Active / To be added / To be cancelled / Not Applicable
    'Medical Insurance Start',   # DD-MM-YYYY
    'Medical Insurance End',     # DD-MM-YYYY
    'Medical Premium Cost',
    # Passport Control — note block (subject / date / note)
    'Passport Control Subject',
    'Passport Control Date',     # DD-MM-YYYY
    'Passport Control Note',
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

        # Guidance note attached to the Badge ID header so it travels with the file.
        from openpyxl.comments import Comment
        note = (
            "Fill Badge ID for each row. Leave any cell blank to keep the existing value.\n"
            "Every column is independent — update any single column on its own.\n"
            "Dates: DD-MM-YYYY (e.g. 24-05-2026).\n"
            "Salary columns are applied for super admins only.\n"
            "The SAMPLE rows below are examples — DELETE them before uploading."
        )
        cmt = Comment(note, "RocketAttendance")
        cmt.width, cmt.height = 340, 170
        ws.cell(row=1, column=1).comment = cmt

        # Two example rows so admins can see the expected format. They use
        # placeholder Badge IDs that won't match real employees, so an accidental
        # upload won't change anyone — but they are meant to be deleted.
        sample_rows = [
            {  # 1) updating salary for an active employee
                'Badge ID': 'SAMPLE-001', 'Name': 'John Doe', 'Email': 'john.doe@example.com',
                'Phone': '0500000000', 'Division (Department)': 'GRC Dep', 'Position': 'Mason',
                'Category': 'Worker', 'Site Name': 'City Walk', 'Sponsor': 'Jafza', 'Employer': 'PIC',
                'Status': 'Active', 'Nationality': 'India', 'Gender': 'Male', 'Marital Status': 'Single',
                'Date of Birth': '15-04-1990', 'Date of Joining': '10-01-2022',
                'Passport Number': 'EF1234567', 'Passport Expiry': '01-03-2028',
                'L.Card/CEC Nr': '94266036', 'MOL ID': '30112058037135',
                'Basic Salary': 1200, 'Accommodation Allowance': 300, 'Transport Allowance': 150,
                'Food Allowance': 100, 'Fixed OT Allowance': 0, 'Other Allowance': 0,
                'Salary Reduction': 0, 'Gross Salary': 1750,
            },
            {  # 2) putting an employee on Leave (note the required leave fields)
                'Badge ID': 'SAMPLE-002', 'Name': 'Jane Smith', 'Status': 'Leave',
                'Leave Approval Date': '20-05-2026', 'Leave Start Date': '24-05-2026',
                'Leave End Date': '10-06-2026', 'Leave Type': 'Annual',
                'Leave Ticket Eligible': 'Eligible', 'Leave Ticket Price': 1500,
            },
        ]
        sample_font = Font(italic=True, color='9CA3AF')
        for r in sample_rows:
            ws.append([r.get(col, '') for col in BULK_EDIT_COLUMNS])
            for col_idx in range(1, len(BULK_EDIT_COLUMNS) + 1):
                ws.cell(row=ws.max_row, column=col_idx).font = sample_font

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
        'Emirates ID': 'emirates_id',
        'Housing Camp': 'camp',
        'Transportation': 'transportation',
        'Working Type': 'working_type',
        'Working Shift': 'working_shift',
        # Insurance (text)
        'WC Insurance Name': 'wc_insurance_name',
        'WC Insurance Status': 'wc_insurance_status',
        'Medical Insurance Name': 'medical_insurance_name',
        'Medical Card Number': 'medical_insurance_card_number',
        'Medical Insurance Status': 'medical_insurance_status',
        # Passport control (text)
        'Passport Control Subject': 'passport_control_subject',
        'Passport Control Note': 'passport_control_note',
    }
    _DATE_FIELD_MAP = {
        'Date of Birth': 'date_of_birth',
        'Date of Joining': 'date_of_joining',
        'Passport Expiry': 'passport_expiry',
        'Visa Expiry': 'visa_expiry_date',
        'Labour Card Expiry': 'labour_card_expiry',
        'Resumption Date': 'resumption_date',
        'Last Working Date': 'last_working_date',
        'Leave Approval Date': 'leave_approval_date',
        'Leave Start Date': 'leave_start_date',
        'Leave End Date': 'leave_end_date',
        # Insurance dates
        'WC Insurance Start': 'wc_insurance_start_date',
        'WC Insurance End': 'wc_insurance_end_date',
        'Medical Insurance Start': 'medical_insurance_start_date',
        'Medical Insurance End': 'medical_insurance_end_date',
        # Passport control date
        'Passport Control Date': 'passport_control_date',
    }
    _DECIMAL_FIELD_MAP = {
        'Basic Salary': 'basic_salary',
        'Accommodation Allowance': 'accommodation_allowance',
        'Transport Allowance': 'transport_allowance',
        'Food Allowance': 'food_allowance',
        'Fixed OT Allowance': 'fixed_ot_allowance',
        'Other Allowance': 'other_allowance',
        'Salary Reduction': 'salary_reduction',
        'Gross Salary': 'gross_salary',
        'Leave Ticket Price': 'leave_ticket_price',
        # Insurance premiums (not salary — applied for all admins)
        'WC Premium Cost': 'wc_insurance_premium_cost',
        'Medical Premium Cost': 'medical_insurance_premium_cost',
    }
    # All salary-component fields are restricted to superusers on import.
    _SALARY_FIELDS = {
        'basic_salary', 'accommodation_allowance', 'transport_allowance',
        'food_allowance', 'fixed_ot_allowance', 'other_allowance',
        'salary_reduction', 'gross_salary',
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

        def get_raw(row, name):
            """Raw cell value (not stringified) — lets coerce_date handle real
            Excel date cells as dates rather than strings."""
            i = col_index.get(name)
            if i is None or i >= len(row):
                return None
            return row[i]

        def coerce_date(s):
            # Real Excel date cells arrive as datetime/date objects — accept those
            # as-is (display format in Excel doesn't matter).
            if s is None:
                return None
            if isinstance(s, datetime):
                return s.date()
            if isinstance(s, date):
                return s
            txt = str(s).strip()
            if not txt:
                return None
            # Typed text dates must be DD-MM-YYYY (or DD/MM/YYYY) — the agreed
            # format. Unambiguous ISO (year-first) is also accepted because that
            # is how Excel serialises a real date cell to text.
            txt = txt.replace('T', ' ').split(' ')[0]
            for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d'):
                try:
                    return datetime.strptime(txt, fmt).date()
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

        # Prefetch every employee referenced in the file in ONE query (instead of
        # one SELECT per row) — big speedup that helps avoid gateway timeouts on
        # large department uploads.
        _all_badges = {
            str(get(r, 'Badge ID')) for r in rows[1:] if get(r, 'Badge ID')
        }
        emp_cache = {
            str(e.badge_number): e
            for e in Employee.objects.filter(badge_number__in=_all_badges)
        }

        TERMINAL_STATUSES = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}
        MASTER_STATUSES = {'Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other'}

        results = {
            'total_rows': 0,
            'updated_count': 0,
            'skipped_count': 0,
            'errors': [],
        }

        # Suspend the per-save FAISS index rebuild during the import. Otherwise
        # every emp.save() triggers a full index rebuild over ALL employees'
        # templates — thousands of rebuilds for a big file, which hangs the
        # worker (504) or OOM-kills it (502). We rebuild ONCE at the end below.
        from . import signals as _signals_mod
        _signals_mod.suspend_employee_index_sync = True
        any_site_changed = False

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
                emp = emp_cache.get(str(badge))
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

                # ── Date fields (each applied independently) ──────────────────
                date_overrides = {}
                for col_name, field_name in self._DATE_FIELD_MAP.items():
                    raw = get_raw(row, col_name)
                    if raw is None or (isinstance(raw, str) and not raw.strip()):
                        continue
                    d = coerce_date(raw)
                    if d is None:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Could not parse date in column "{col_name}": "{raw}". Use DD-MM-YYYY.',
                        })
                        raise ValueError('skip')
                    date_overrides[field_name] = d
                    setattr(emp, field_name, d)   # apply immediately — no column is linked

                # ── Decimal fields ───────────────────────────────────────────
                # Salary fields only applied for superuser
                for col_name, field_name in self._DECIMAL_FIELD_MAP.items():
                    raw = get(row, col_name)
                    if raw is None:
                        continue
                    if field_name in self._SALARY_FIELDS and not request.user.is_superuser:
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

                # ── Leave Type (independent) ─────────────────────────────────
                l_typ = get(row, 'Leave Type')
                if l_typ is not None:
                    emp.leave_type = l_typ

                # ── Termination Reason (independent) ─────────────────────────
                t_rsn = get(row, 'Termination Reason')
                if t_rsn is not None:
                    emp.termination_reason = t_rsn

                # ── Status (independent) ─────────────────────────────────────
                # Every column updates on its own. Changing Status no longer
                # requires — or clears — any other column. The date/leave/reason
                # columns above are already applied individually; here we only
                # validate the status value and log the transition for history.
                new_status = get(row, 'Status')
                if new_status is not None:
                    if new_status not in MASTER_STATUSES:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Invalid Status "{new_status}". Must be one of: {", ".join(sorted(MASTER_STATUSES))}',
                        })
                        raise ValueError('skip')

                    old_status = emp.status
                    emp.status = new_status

                    if old_status != new_status:
                        EmployeeStatusHistory.objects.create(
                            employee=emp,
                            old_status=old_status,
                            new_status=new_status,
                            leave_approval_date=date_overrides.get('leave_approval_date'),
                            leave_start_date=date_overrides.get('leave_start_date'),
                            leave_end_date=date_overrides.get('leave_end_date'),
                            resumption_date=date_overrides.get('resumption_date'),
                            last_working_date=date_overrides.get('last_working_date'),
                            leave_type=l_typ,
                            note=(
                                f"{old_status or '—'} → {new_status} (bulk edit)"
                                + (f" — Reason: {t_rsn}" if t_rsn else '')
                            ),
                            changed_by=request.user if request.user.is_authenticated else None,
                        )

                # ── Site change history ──────────────────────────────────────
                site_changed = (
                    (old_site and emp.site and old_site.id != emp.site_id)
                    or (old_site and not emp.site)
                    or (not old_site and emp.site)
                )
                if site_changed:
                    any_site_changed = True
                    eff_raw = get_raw(row, 'Site Effective From')
                    eff_from = coerce_date(eff_raw)
                    if eff_raw not in (None, '') and not eff_from:
                        results['errors'].append({
                            'row': row_idx, 'badge_number': badge,
                            'error': f'Could not parse "Site Effective From": "{eff_raw}". Use DD-MM-YYYY.',
                        })
                        raise ValueError('skip')
                    # Optional now — default to today so Site Name can be changed on its own.
                    if not eff_from:
                        eff_from = date.today()
                    EmployeeSiteHistory.objects.create(
                        employee=emp,
                        old_site=old_site,
                        new_site=emp.site,
                        effective_from=eff_from,
                        note=(
                            f"{old_site.name if old_site else '—'} → "
                            f"{emp.site.name if emp.site else '—'} (bulk edit)"
                        ),
                        working_type=emp.working_type,
                        working_shift=emp.working_shift,
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

        # Re-enable index sync and rebuild the FAISS index ONCE (only needed if a
        # site assignment changed — that's the only thing affecting the partition).
        _signals_mod.suspend_employee_index_sync = False
        if any_site_changed:
            try:
                qs = FaceTemplate.objects.all().select_related('employee').only(
                    "id", "employee_id", "embedding", "employee__site_id"
                )
                tuples = [
                    (t.id, t.employee_id, t.employee.site_id, np.array(t.embedding, dtype=np.float32))
                    for t in qs
                ]
                ENGINE.rebuild_index(tuples)
            except Exception:  # noqa: BLE001
                logger.exception("FAISS rebuild after bulk edit failed")

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
                visa_expiry_date=parse_date(data.get('visa_expiry_date')),
                labor_card_number=data.get('labor_card_number'),
                mol_id=data.get('mol_id'),
                job_description=data.get('job_description'),
                sponsor=data.get('sponsor'),
                employer=(data.get('employer') if data.get('employer') in EMPLOYER_CHOICES else None),
                site=site,
                gross_salary=parse_decimal(data.get('gross_salary')),
                basic_salary=parse_decimal(data.get('basic_salary')),
                accommodation_allowance=parse_decimal(data.get('accommodation_allowance')),
                transport_allowance=parse_decimal(data.get('transport_allowance')),
                food_allowance=parse_decimal(data.get('food_allowance')),
                fixed_ot_allowance=parse_decimal(data.get('fixed_ot_allowance')),
                other_allowance=parse_decimal(data.get('other_allowance')),
                salary_reduction=parse_decimal(data.get('salary_reduction')),
                salary_remarks=data.get('salary_remarks') or None,
                category=data.get('category', 'worker'),
                camp=data.get('camp'),
                transportation=data.get('transportation')
            )

            # Record initial salary snapshot when any salary component is provided
            if any([new_emp.basic_salary, new_emp.gross_salary, new_emp.accommodation_allowance,
                    new_emp.transport_allowance, new_emp.food_allowance, new_emp.fixed_ot_allowance,
                    new_emp.other_allowance, new_emp.salary_reduction]):
                EmployeeSalaryHistory.objects.create(
                    employee=new_emp,
                    basic_salary=new_emp.basic_salary,
                    accommodation_allowance=new_emp.accommodation_allowance,
                    transport_allowance=new_emp.transport_allowance,
                    food_allowance=new_emp.food_allowance,
                    fixed_ot_allowance=new_emp.fixed_ot_allowance,
                    other_allowance=new_emp.other_allowance,
                    salary_reduction=new_emp.salary_reduction,
                    gross_salary=new_emp.gross_salary,
                    remarks=new_emp.salary_remarks or 'Initial salary on employee creation',
                    effective_from=parse_date(data.get('date_of_joining')) or timezone.localdate(),
                    changed_by=request.user if request.user.is_authenticated else None,
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
                'visa_expiry_date': str(emp.visa_expiry_date) if emp.visa_expiry_date else '',
                'labor_card_number': emp.labor_card_number,
                'mol_id': emp.mol_id,
                'job_description': emp.job_description,
                'sponsor': emp.sponsor or '',
                'employer': emp.employer or '',
                'site': emp.site.id if emp.site else '',
                'site_name': emp.site.name if emp.site else '',
                'camp': emp.camp,
                'transportation': emp.transportation,
                # Document URLs (frontend uses these for "View" buttons)
                'passport_document_url': emp.passport_document.url if emp.passport_document else '',
                'visa_document_url': emp.visa_document.url if emp.visa_document else '',
                'labour_card_document_url': emp.labour_card_document.url if emp.labour_card_document else '',
                # Insurance & End-of-Service
                'wc_insurance_name': emp.wc_insurance_name or '',
                'wc_insurance_start_date': str(emp.wc_insurance_start_date) if emp.wc_insurance_start_date else '',
                'wc_insurance_end_date': str(emp.wc_insurance_end_date) if emp.wc_insurance_end_date else '',
                'wc_insurance_status': emp.wc_insurance_status or '',
                'wc_insurance_premium_cost': str(emp.wc_insurance_premium_cost) if emp.wc_insurance_premium_cost is not None else '',
                'medical_insurance_name': emp.medical_insurance_name or '',
                'medical_insurance_start_date': str(emp.medical_insurance_start_date) if emp.medical_insurance_start_date else '',
                'medical_insurance_end_date': str(emp.medical_insurance_end_date) if emp.medical_insurance_end_date else '',
                'medical_insurance_card_number': emp.medical_insurance_card_number or '',
                'medical_insurance_status': emp.medical_insurance_status or '',
                'medical_insurance_premium_cost': str(emp.medical_insurance_premium_cost) if emp.medical_insurance_premium_cost is not None else '',
                'eos_subject': emp.eos_subject or '',
                'eos_date': str(emp.eos_date) if emp.eos_date else '',
                'eos_note': emp.eos_note or '',
                'emirates_id': emp.emirates_id or '',
                'working_type': emp.working_type or '',
                'working_shift': emp.working_shift or '',
                'labour_card_expiry': str(emp.labour_card_expiry) if emp.labour_card_expiry else '',
                'passport_control_subject': emp.passport_control_subject or '',
                'passport_control_date': str(emp.passport_control_date) if emp.passport_control_date else '',
                'passport_control_note': emp.passport_control_note or '',
            }
            if request.user.is_superuser:
                data['gross_salary'] = str(emp.gross_salary) if emp.gross_salary else ''
                data['basic_salary'] = str(emp.basic_salary) if emp.basic_salary else ''
                data['accommodation_allowance'] = str(emp.accommodation_allowance) if emp.accommodation_allowance is not None else ''
                data['transport_allowance'] = str(emp.transport_allowance) if emp.transport_allowance is not None else ''
                data['food_allowance'] = str(emp.food_allowance) if emp.food_allowance is not None else ''
                data['fixed_ot_allowance'] = str(emp.fixed_ot_allowance) if emp.fixed_ot_allowance is not None else ''
                data['other_allowance'] = str(emp.other_allowance) if emp.other_allowance is not None else ''
                data['salary_reduction'] = str(emp.salary_reduction) if emp.salary_reduction is not None else ''
                data['salary_remarks'] = emp.salary_remarks or ''
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
            if 'emirates_id' in data:
                emp.emirates_id = (data.get('emirates_id') or '').strip() or None
            if 'working_type' in data:
                emp.working_type = (data.get('working_type') or '').strip() or None
            if 'working_shift' in data:
                emp.working_shift = (data.get('working_shift') or '').strip() or None
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
                    working_type=emp.working_type,
                    working_shift=emp.working_shift,
                    changed_by=request.user if request.user.is_authenticated else None,
                )

            emp.date_of_birth = parse_date(data.get('date_of_birth'))
            emp.date_of_joining = parse_date(data.get('date_of_joining'))
            emp.passport_expiry = parse_date(data.get('passport_expiry'))
            emp.visa_expiry_date = parse_date(data.get('visa_expiry_date'))

            if request.user.is_superuser:
                emp.gross_salary = parse_decimal(data.get('gross_salary'))
                emp.basic_salary = parse_decimal(data.get('basic_salary'))
                # Salary components — only overwrite when the request actually
                # includes the key, so saving an unrelated field (e.g., status)
                # doesn't wipe components that aren't on the current form view.
                for k in ('accommodation_allowance', 'transport_allowance',
                          'food_allowance', 'fixed_ot_allowance',
                          'other_allowance', 'salary_reduction'):
                    if k in data:
                        setattr(emp, k, parse_decimal(data.get(k)))
                if 'salary_remarks' in data:
                    emp.salary_remarks = data.get('salary_remarks') or None
                emp.salary_grade = data.get('salary_grade', emp.salary_grade)

            # ── Insurance (WC + Medical) and End-of-Service ──────────────────
            def _txt(k):
                return (data.get(k) or '').strip() or None
            if 'wc_insurance_name' in data:         emp.wc_insurance_name = _txt('wc_insurance_name')
            if 'wc_insurance_start_date' in data:   emp.wc_insurance_start_date = parse_date(data.get('wc_insurance_start_date'))
            if 'wc_insurance_end_date' in data:     emp.wc_insurance_end_date = parse_date(data.get('wc_insurance_end_date'))
            if 'wc_insurance_status' in data:       emp.wc_insurance_status = _txt('wc_insurance_status')
            if 'wc_insurance_premium_cost' in data: emp.wc_insurance_premium_cost = parse_decimal(data.get('wc_insurance_premium_cost'))
            if 'medical_insurance_name' in data:         emp.medical_insurance_name = _txt('medical_insurance_name')
            if 'medical_insurance_start_date' in data:   emp.medical_insurance_start_date = parse_date(data.get('medical_insurance_start_date'))
            if 'medical_insurance_end_date' in data:     emp.medical_insurance_end_date = parse_date(data.get('medical_insurance_end_date'))
            if 'medical_insurance_card_number' in data:  emp.medical_insurance_card_number = _txt('medical_insurance_card_number')
            if 'medical_insurance_status' in data:       emp.medical_insurance_status = _txt('medical_insurance_status')
            if 'medical_insurance_premium_cost' in data: emp.medical_insurance_premium_cost = parse_decimal(data.get('medical_insurance_premium_cost'))
            if 'eos_subject' in data: emp.eos_subject = _txt('eos_subject')
            if 'eos_date' in data:    emp.eos_date = parse_date(data.get('eos_date'))
            if 'eos_note' in data:    emp.eos_note = _txt('eos_note')

            # ── Labour card expiry + Passport Control note ───────────────────
            if 'labour_card_expiry' in data:       emp.labour_card_expiry = parse_date(data.get('labour_card_expiry'))
            if 'passport_control_subject' in data: emp.passport_control_subject = _txt('passport_control_subject')
            if 'passport_control_date' in data:    emp.passport_control_date = parse_date(data.get('passport_control_date'))
            if 'passport_control_note' in data:    emp.passport_control_note = _txt('passport_control_note')

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
                'working_type': h.working_type,
                'working_shift': h.working_shift,
            })

        return Response({
            'employee_id': emp.id,
            'employee_name': emp.name,
            'current_site': emp.site.name if emp.site else None,
            'current_site_id': emp.site.id if emp.site else None,
            'segments': segments,
        })


class EmployeeSalaryHistoryView(APIView):
    """List or append salary-change snapshots for an employee.

    GET  /api/attendance/employees/<id>/salary-history/  → current + history
    POST /api/attendance/employees/<id>/salary-history/  → append an increment.
        Body (any subset, plus optional 'remarks' and 'effective_from'):
            basic_salary, accommodation_allowance, transport_allowance,
            food_allowance, fixed_ot_allowance, other_allowance,
            salary_reduction, gross_salary
        Empty / missing fields keep their current value on the Employee row.
    """
    permission_classes = [IsAdminUser]

    _COMPONENTS = (
        'basic_salary',
        'accommodation_allowance',
        'transport_allowance',
        'food_allowance',
        'fixed_ot_allowance',
        'other_allowance',
        'salary_reduction',
        'gross_salary',
    )

    @staticmethod
    def _dec(v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            from decimal import Decimal
            return Decimal(s.replace(',', ''))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _str(v):
        return str(v) if v is not None else None

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        rows = (
            emp.salary_history
               .select_related('changed_by')
               .order_by('-effective_from', '-changed_at')
        )
        history = []
        for h in rows:
            history.append({
                'id': h.id,
                'basic_salary': self._str(h.basic_salary),
                'accommodation_allowance': self._str(h.accommodation_allowance),
                'transport_allowance': self._str(h.transport_allowance),
                'food_allowance': self._str(h.food_allowance),
                'fixed_ot_allowance': self._str(h.fixed_ot_allowance),
                'other_allowance': self._str(h.other_allowance),
                'salary_reduction': self._str(h.salary_reduction),
                'gross_salary': self._str(h.gross_salary),
                'remarks': h.remarks or '',
                'effective_from': str(h.effective_from) if h.effective_from else None,
                'changed_at': h.changed_at.isoformat() if h.changed_at else None,
                'changed_by': h.changed_by.username if h.changed_by else None,
            })

        return Response({
            'employee_id': emp.id,
            'current': {
                'basic_salary': self._str(emp.basic_salary),
                'accommodation_allowance': self._str(emp.accommodation_allowance),
                'transport_allowance': self._str(emp.transport_allowance),
                'food_allowance': self._str(emp.food_allowance),
                'fixed_ot_allowance': self._str(emp.fixed_ot_allowance),
                'other_allowance': self._str(emp.other_allowance),
                'salary_reduction': self._str(emp.salary_reduction),
                'gross_salary': self._str(emp.gross_salary),
                'remarks': emp.salary_remarks or '',
            },
            'count': len(history),
            'history': history,
        })

    def post(self, request, employee_id):
        from datetime import datetime as _dt

        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        data = request.data
        provided = {}
        for k in self._COMPONENTS:
            raw = data.get(k)
            if raw is None or str(raw).strip() == '':
                continue
            v = self._dec(raw)
            if v is None:
                return Response({'error': f'Could not parse number for "{k}".'}, status=400)
            provided[k] = v

        if not provided:
            return Response(
                {'error': 'Provide at least one salary component to record an increment.'},
                status=400,
            )

        eff_from_raw = (data.get('effective_from') or '').strip()
        if eff_from_raw:
            try:
                effective_from = _dt.strptime(eff_from_raw, '%Y-%m-%d').date()
            except ValueError:
                return Response({'error': 'effective_from must be YYYY-MM-DD.'}, status=400)
        else:
            effective_from = timezone.localdate()

        remarks = (data.get('remarks') or '').strip() or None

        # Apply changes to the live Employee row, then snapshot.
        for k, v in provided.items():
            setattr(emp, k, v)
        emp.save()

        snap = EmployeeSalaryHistory.objects.create(
            employee=emp,
            basic_salary=emp.basic_salary,
            accommodation_allowance=emp.accommodation_allowance,
            transport_allowance=emp.transport_allowance,
            food_allowance=emp.food_allowance,
            fixed_ot_allowance=emp.fixed_ot_allowance,
            other_allowance=emp.other_allowance,
            salary_reduction=emp.salary_reduction,
            gross_salary=emp.gross_salary,
            remarks=remarks,
            effective_from=effective_from,
            changed_by=request.user if request.user.is_authenticated else None,
        )
        return Response({
            'success': True,
            'id': snap.id,
            'effective_from': str(snap.effective_from),
        }, status=201)


class EmployeeAttachmentsView(APIView):
    """List, add, or remove free-form attachments for an employee.

    GET    /api/attendance/employees/<id>/attachments/                 → list
    POST   /api/attendance/employees/<id>/attachments/  (multipart)    → add: name, file
    DELETE /api/attendance/employees/<id>/attachments/?id=<attachment> → remove one
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        rows = (
            emp.attachments
               .select_related('uploaded_by')
               .order_by('-uploaded_at')
        )
        items = [{
            'id': a.id,
            'name': a.name,
            'file_url': a.file.url if a.file else None,
            'uploaded_at': a.uploaded_at.isoformat() if a.uploaded_at else None,
            'uploaded_by': a.uploaded_by.username if a.uploaded_by else None,
        } for a in rows]
        return Response({
            'employee_id': emp.id,
            'count': len(items),
            'attachments': items,
        })

    def post(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)

        name = (request.data.get('name') or '').strip()
        file_obj = request.FILES.get('file')

        if not name:
            return Response({'error': 'Attachment name is required.'}, status=400)
        if not file_obj:
            return Response({'error': 'Please choose a file to upload.'}, status=400)

        a = EmployeeAttachment.objects.create(
            employee=emp,
            name=name[:200],
            file=file_obj,
            uploaded_by=request.user if request.user.is_authenticated else None,
        )
        return Response({
            'success': True,
            'id': a.id,
            'name': a.name,
            'file_url': a.file.url if a.file else None,
            'uploaded_at': a.uploaded_at.isoformat(),
            'uploaded_by': a.uploaded_by.username if a.uploaded_by else None,
        }, status=201)

    def delete(self, request, employee_id):
        att_id = request.query_params.get('id') or request.data.get('id')
        if not att_id:
            return Response({'error': 'Attachment id is required.'}, status=400)
        try:
            a = EmployeeAttachment.objects.get(id=att_id, employee_id=employee_id)
        except EmployeeAttachment.DoesNotExist:
            return Response({'error': 'Attachment not found'}, status=404)
        if a.file:
            a.file.delete(save=False)
        a.delete()
        return Response({'success': True})


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

            # Filter by Site/Position/Department/Category/Status/Employer (Query Param or Admin Profile)
            site_id = request.GET.get('site')
            position_filter = request.GET.get('position')
            department_filter = request.GET.get('department')
            category_filter = request.GET.get('category')  # now strictly Employee.category enum (staff/worker)
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

            # Position filter (trade name) — matches Employee.position first, with
            # legacy salary_grade as a fall-back for employees still on the old field.
            if position_filter and position_filter != 'all':
                employees = employees.filter(
                    Q(position__iexact=position_filter) |
                    (Q(position__in=['', None]) & Q(salary_grade__iexact=position_filter))
                )
                attendance = attendance.filter(
                    Q(user__position__iexact=position_filter) |
                    (Q(user__position__in=['', None]) & Q(user__salary_grade__iexact=position_filter))
                )

            # Department filter — strict ilike match on Employee.department
            if department_filter and department_filter != 'all':
                employees = employees.filter(department__iexact=department_filter)
                attendance = attendance.filter(user__department__iexact=department_filter)

            # Category filter — Staff / Worker enum on Employee.category
            if category_filter and category_filter != 'all':
                employees = employees.filter(category__iexact=category_filter)
                attendance = attendance.filter(user__category__iexact=category_filter)

            # Snapshot the queryset BEFORE the status filter so the breakdown
            # counts stay accurate even when the admin has clicked a pill — clicking
            # "Active" shouldn't zero out the Leave / Resigned / etc. counters.
            employees_pre_status = employees

            if status_filter and status_filter != 'all':
                employees = employees.filter(status__iexact=status_filter)
                attendance = attendance.filter(user__status__iexact=status_filter)

            if employer_filter and employer_filter != 'all':
                employees = employees.filter(employer__iexact=employer_filter)
                attendance = attendance.filter(user__employer__iexact=employer_filter)
                employees_pre_status = employees_pre_status.filter(employer__iexact=employer_filter)

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

            # Breakdown by employment status — one bucket per master status, plus
            # a catch-all 'Other' for anything outside the standard list (legacy
            # imports etc.). Counts honour site/category/employer filters but
            # NOT the status filter — otherwise clicking the Active pill would
            # zero out every other pill and trap the admin on one bucket.
            _MASTER = ['Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other']
            status_counts = {s: 0 for s in _MASTER}
            for raw_status in employees_pre_status.values_list('status', flat=True):
                key = (raw_status or '').strip() or 'Other'
                if key not in status_counts:
                    key = 'Other'
                status_counts[key] += 1

            # Absent Today = Active employees that didn't post any attendance
            # record today. Counts within the same filter scope as the rest of
            # the stats, so site/department/etc. narrow it down too.
            today_user_ids = list(attendance.values_list('user_id', flat=True).distinct())
            absent_today_count = (
                employees.filter(status__iexact='Active')
                         .exclude(id__in=today_user_ids)
                         .count()
            )

            return Response(
                {
                    "total_employees": employees.count(),
                    "today_attendance_count": attendance.count(),
                    "late_count": attendance.filter(status='late').count(),
                    "absent_today_count": absent_today_count,
                    "total_sites": all_sites.count(),
                    "sites": [{"id": s.id, "name": s.name} for s in all_sites],
                    "categories": unique_categories,
                    "statuses": unique_statuses,
                    "status_counts": status_counts,
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
        # Accept ?date=YYYY-MM-DD (defaults to today). Future dates are clamped
        # back to today since attendance can't be in the future.
        date_str = (request.GET.get('date') or '').strip()
        target_date = timezone.localdate()
        if date_str:
            try:
                from datetime import datetime as _dt
                parsed = _dt.strptime(date_str, '%Y-%m-%d').date()
                if parsed <= timezone.localdate():
                    target_date = parsed
            except ValueError:
                pass

        site_id = request.GET.get('site')

        # ── Pool 1: Out-of-bounds (geofence failure) ─────────────────────────
        geofence_alerts = Attendance.objects.filter(
            date=target_date, is_within_geofence=False,
        ).select_related('user', 'user__site')

        # ── Pool 2: Employee on Leave but face-scanned on the target date ───
        # An employee marked an attendance while the system has them on Leave —
        # admin needs to know either to revoke leave or correct the attendance.
        on_leave_alerts = (
            Attendance.objects
            .filter(date=target_date, user__status='Leave')
            .filter(
                Q(check_in_time__isnull=False) | Q(check_out_time__isnull=False)
            )
            .filter(
                Q(user__leave_start_date__isnull=True)
                | Q(user__leave_start_date__lte=target_date)
            )
            .filter(
                Q(user__leave_end_date__isnull=True)
                | Q(user__leave_end_date__gte=target_date)
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
                "site_id": a.user.site.id if a.user.site else None,
                "badge_number": a.user.badge_number or "-",
                "position": a.user.position or "-",
                "department": a.user.department or "-",
                "time": timezone.localtime(a.check_in_time).strftime("%H:%M") if a.check_in_time else (timezone.localtime(a.check_out_time).strftime("%H:%M") if a.check_out_time else "-"),
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
                "site_id": a.user.site.id if a.user.site else None,
                "badge_number": a.user.badge_number or "-",
                "position": a.user.position or "-",
                "department": a.user.department or "-",
                "time": timezone.localtime(a.check_in_time).strftime("%H:%M") if a.check_in_time else (timezone.localtime(a.check_out_time).strftime("%H:%M") if a.check_out_time else "-"),
                "lat": a.latitude,
                "long": a.longitude,
                "status": "Marked attendance while on Leave",
                "kind": "on_leave",
            })
        # Wrap in an object so the date used can be echoed back to the UI.
        return Response({
            'date': str(target_date),
            'alerts': data,
        })


class AttendanceAlertsExportView(APIView):
    """Excel export of every employee that triggered an alert on the chosen
    date (geofence-out-of-bounds OR marked-while-on-leave). Honours all the
    same filters as the alerts grid: site / position / department / category /
    employer + the date picker.

    Format = the full exhaustive employee export (same column set as
    "Export Filtered" / "Download Selected"), plus three alert-context columns
    in front: Alert Type, Alert Date, Alert Time.

    GET /api/attendance/alerts/export/?date=YYYY-MM-DD&site=…&status=…&position=…&department=…&category=…&employer=…
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from datetime import datetime as _dt
        from django.http import HttpResponse
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, PatternFill, Alignment

        date_str = (request.GET.get('date') or '').strip()
        target_date = timezone.localdate()
        if date_str:
            try:
                parsed = _dt.strptime(date_str, '%Y-%m-%d').date()
                if parsed <= timezone.localdate():
                    target_date = parsed
            except ValueError:
                pass

        site_id           = request.GET.get('site')
        status_filter     = request.GET.get('status')
        position_filter   = request.GET.get('position')
        department_filter = request.GET.get('department')
        category_filter   = request.GET.get('category')  # Staff / Worker enum
        employer_filter   = request.GET.get('employer')

        # ── Same two pools as AttendanceAlertsView ─────────────────────────
        geofence_qs = (Attendance.objects
                       .filter(date=target_date, is_within_geofence=False)
                       .select_related('user', 'user__site'))
        on_leave_qs = (Attendance.objects
                       .filter(date=target_date, user__status='Leave')
                       .filter(Q(check_in_time__isnull=False) | Q(check_out_time__isnull=False))
                       .filter(Q(user__leave_start_date__isnull=True)
                               | Q(user__leave_start_date__lte=target_date))
                       .filter(Q(user__leave_end_date__isnull=True)
                               | Q(user__leave_end_date__gte=target_date))
                       .select_related('user', 'user__site'))

        # ── Site-admin scope ───────────────────────────────────────────────
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                assigned_sites = profile.sites.all()
                if site_id and site_id != 'all':
                    if not assigned_sites.filter(id=site_id).exists():
                        geofence_qs = geofence_qs.none()
                        on_leave_qs = on_leave_qs.none()
                    else:
                        geofence_qs = geofence_qs.filter(user__site_id=site_id)
                        on_leave_qs = on_leave_qs.filter(user__site_id=site_id)
                else:
                    geofence_qs = geofence_qs.filter(user__site__in=assigned_sites)
                    on_leave_qs = on_leave_qs.filter(user__site__in=assigned_sites)
            except AdminProfile.DoesNotExist:
                geofence_qs = geofence_qs.none()
                on_leave_qs = on_leave_qs.none()
        elif site_id and site_id != 'all':
            geofence_qs = geofence_qs.filter(user__site_id=site_id)
            on_leave_qs = on_leave_qs.filter(user__site_id=site_id)

        # ── Mirror the dashboard's other top-bar filters ───────────────────
        if status_filter and status_filter != 'all':
            geofence_qs = geofence_qs.filter(user__status__iexact=status_filter)
            on_leave_qs = on_leave_qs.filter(user__status__iexact=status_filter)
        if position_filter and position_filter != 'all':
            geofence_qs = geofence_qs.filter(
                Q(user__position__iexact=position_filter) |
                (Q(user__position__in=['', None]) & Q(user__salary_grade__iexact=position_filter))
            )
            on_leave_qs = on_leave_qs.filter(
                Q(user__position__iexact=position_filter) |
                (Q(user__position__in=['', None]) & Q(user__salary_grade__iexact=position_filter))
            )
        if department_filter and department_filter != 'all':
            geofence_qs = geofence_qs.filter(user__department__iexact=department_filter)
            on_leave_qs = on_leave_qs.filter(user__department__iexact=department_filter)
        if category_filter and category_filter != 'all':
            geofence_qs = geofence_qs.filter(user__category__iexact=category_filter)
            on_leave_qs = on_leave_qs.filter(user__category__iexact=category_filter)
        if employer_filter and employer_filter != 'all':
            geofence_qs = geofence_qs.filter(user__employer__iexact=employer_filter)
            on_leave_qs = on_leave_qs.filter(user__employer__iexact=employer_filter)

        # ── Build one row per alerted attendance, dedup geofence ∩ leave ──
        seen_ids = set()
        rows = []  # list of (alert_type, alert_time_str, employee)
        for a in geofence_qs:
            seen_ids.add(a.id)
            t = a.check_in_time or a.check_out_time
            rows.append(('Out of Bounds',
                         t.strftime('%H:%M:%S') if t else '-',
                         a.user))
        for a in on_leave_qs:
            if a.id in seen_ids:
                continue
            t = a.check_in_time or a.check_out_time
            rows.append(('Marked attendance while on Leave',
                         t.strftime('%H:%M:%S') if t else '-',
                         a.user))

        # ── Workbook: 3 alert columns + every Employee column ──────────────
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f'Alerts {target_date}'

        alert_headers = ['Alert Type', 'Alert Date', 'Alert Time']
        emp_headers = [h for h, _ in EMPLOYEE_EXPORT_COLUMNS]
        headers = alert_headers + emp_headers
        ws.append(headers)

        header_fill = PatternFill(start_color='B91C1C', end_color='B91C1C', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')
        center = Alignment(horizontal='center', vertical='center', wrap_text=True)
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center

        for alert_type, alert_time, emp in rows:
            row_values = [alert_type, str(target_date), alert_time]
            row_values += [extractor(emp) for _, extractor in EMPLOYEE_EXPORT_COLUMNS]
            ws.append(row_values)

        # Column widths
        for col in range(1, len(headers) + 1):
            letter = get_column_letter(col)
            name = headers[col - 1].lower()
            if any(k in name for k in ('name', 'description', 'remarks', 'reason', 'type')):
                ws.column_dimensions[letter].width = 32
            else:
                ws.column_dimensions[letter].width = 18
        ws.freeze_panes = 'A2'

        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=geofence_alerts_{target_date}.xlsx'
        wb.save(response)
        return response


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

        # Filter by position / department / category
        position_filter = request.GET.get('position')
        department_filter = request.GET.get('department')
        category_filter = request.GET.get('category')

        if position_filter and position_filter != 'all':
            employees = employees.filter(
                Q(position__iexact=position_filter) |
                (Q(position__in=['', None]) & Q(salary_grade__iexact=position_filter))
            )
        if department_filter and department_filter != 'all':
            employees = employees.filter(department__iexact=department_filter)
        if category_filter and category_filter != 'all':
            employees = employees.filter(category__iexact=category_filter)

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

        # The huge field is face_embedding (~512 floats/employee). Only the offline
        # face-sync (has_face=true) or an explicit light=0 actually needs it; every
        # other caller — including OLD app builds that pull the whole list — now
        # gets the light payload, so responses drop from ~8 MB to a few hundred KB.
        _light_raw = str(request.GET.get('light', '')).lower()
        include_embedding = (request.GET.get('has_face') == 'true') or (_light_raw in ('0', 'false', 'no'))
        use_light = not include_embedding

        # Pagination — cap list pages hard so no single request can pull thousands
        # of rows (old apps sent no per_page → 1000). Embedding/sync callers keep
        # their requested page size so paginated sync still works.
        try:
            per_page = int(request.GET.get('per_page', 50))
        except (ValueError, TypeError):
            per_page = 50
        if not include_embedding:
            per_page = min(per_page, 100)
        paginator = PageNumberPagination()
        paginator.page_size = max(1, per_page)

        if use_light:
            employees = employees.defer('face_embedding')

        # Attendance Filter (Present / Late / Absent) — applied after other
        # filters but before pagination so the table reflects the KPI card
        # the admin clicked.
        attendance_filter = request.GET.get('attendance_filter')
        if attendance_filter in ('present', 'late'):
            today = timezone.localdate()
            attendance_qs = Attendance.objects.filter(date=today)
            if attendance_filter == 'late':
                attendance_qs = attendance_qs.filter(status='late')
            else:
                attendance_qs = attendance_qs.filter(status='present')
            present_employee_ids = attendance_qs.values_list('user_id', flat=True)
            employees = employees.filter(id__in=present_employee_ids)
        elif attendance_filter == 'absent':
            # Active employees that didn't post any attendance today.
            today = timezone.localdate()
            today_ids = Attendance.objects.filter(date=today).values_list('user_id', flat=True).distinct()
            employees = employees.filter(status__iexact='Active').exclude(id__in=today_ids)

        result_page = paginator.paginate_queryset(employees, request)

        serializer_cls = EmployeeSerializer if include_embedding else EmployeeListLightSerializer
        serializer = serializer_cls(
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
    status_filter = request.GET.get('status', 'all') # 'enrolled', 'not_enrolled', 'all' (face enrollment)
    emp_status_filter = request.GET.get('emp_status', 'all')  # employment status (Active / Leave / etc.)
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

    # Employment-status breakdown — computed BEFORE the emp_status filter so the
    # admin can always see the per-status totals within the site / search / face
    # context and use them as quick filters.
    _MASTER = ['Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding', 'Other']
    status_counts = {s: 0 for s in _MASTER}
    for raw_status in employees.values_list('status', flat=True):
        key = (raw_status or '').strip() or 'Other'
        if key not in status_counts:
            key = 'Other'
        status_counts[key] += 1

    # Employment status quick-filter (Active / Leave / Resigned / …)
    if emp_status_filter and emp_status_filter != 'all':
        if emp_status_filter == 'Other':
            employees = employees.exclude(status__in=_MASTER[:-1])  # everything not in the 6 named ones
        else:
            employees = employees.filter(status__iexact=emp_status_filter)

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

        # status_counts was computed earlier (before the emp_status filter)
        # so the breakdown shows totals within the site/search/face context.

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
                'emp_status': emp_status_filter,
                'search': search_query,
                'per_page': per_page
            },
            'sites': list(Site.objects.values('id', 'name')) if is_superuser else [],
            'permissions': {
                'is_superuser': is_superuser
            },
            'total_count': paginator.count,
            'status_counts': status_counts,
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

            # Resolve leave window — prefer the dates stored on the Employee row, but
            # fall back to the most recent EmployeeStatusHistory entry with
            # new_status='Leave'. This covers employees set to Leave via the Excel
            # import (which sets `status` but doesn't always populate the dates).
            _lv_approval = employee.leave_approval_date
            _lv_start    = employee.leave_start_date
            _lv_end      = employee.leave_end_date
            _lv_type     = employee.leave_type
            _lv_eligible = employee.leave_ticket_eligible
            _lv_price    = employee.leave_ticket_price
            if employee.status == 'Leave' and not (_lv_approval or _lv_start or _lv_end):
                _last_leave_hist = (
                    EmployeeStatusHistory.objects
                    .filter(employee=employee, new_status='Leave')
                    .order_by('-changed_at')
                    .first()
                )
                if _last_leave_hist:
                    _lv_approval = _lv_approval or _last_leave_hist.leave_approval_date
                    _lv_start    = _lv_start    or _last_leave_hist.leave_start_date
                    _lv_end      = _lv_end      or _last_leave_hist.leave_end_date
                    _lv_type     = _lv_type     or _last_leave_hist.leave_type
                    if _lv_eligible is None:
                        _lv_eligible = _last_leave_hist.leave_ticket_eligible
                    if _lv_price is None:
                        _lv_price = _last_leave_hist.leave_ticket_price

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
                    # Leave dates: from Employee.leave_*, with EmployeeStatusHistory fallback.
                    'leave_approval_date': str(_lv_approval) if _lv_approval else None,
                    'leave_start_date':    str(_lv_start)    if _lv_start    else None,
                    'leave_end_date':      str(_lv_end)      if _lv_end      else None,
                    'leave_type':          _lv_type          or None,
                    'leave_ticket_eligible': (
                        None if _lv_eligible is None
                        else ('Eligible' if _lv_eligible else 'Not Eligible')
                    ),
                    'leave_ticket_price':  str(_lv_price) if _lv_price is not None else None,
                    'leave_total_days': (
                        (_lv_end - _lv_start).days + 1
                        if _lv_start and _lv_end else None
                    ),
                    'leave_dates_missing': (
                        employee.status == 'Leave'
                        and not (_lv_approval or _lv_start or _lv_end)
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
                    'visa_expiry_date': str(employee.visa_expiry_date) if employee.visa_expiry_date else None,
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
                    # Only show a location pin when this punch actually happened.
                    "latitude": record.latitude if (record and record.check_in_time) else None,
                    "longitude": record.longitude if (record and record.check_in_time) else None,
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
                    # Only show a location pin when this punch actually happened.
                    "latitude": record.latitude if (record and record.check_out_time) else None,
                    "longitude": record.longitude if (record and record.check_out_time) else None,
                }
                data['slots'] = slots
                # Expose the record id + flags so the detail page can offer to
                # fill in a missing check-in (checked out without checking in).
                data['attendance_id'] = record.id if record else None
                data['missing_check_in'] = bool(
                    record and record.check_out_time and not record.check_in_time
                )
                data['can_edit_attendance'] = bool(request.user.is_superuser)
                # Day-specific site assignment — used by the frontend to overlay the
                # *correct* geofence polygon for this day, even if the employee has
                # since been re-assigned to a different site.
                _day_site = None
                for h in employee.site_history.select_related('new_site', 'old_site').order_by('effective_from'):
                    if h.effective_from <= current_date:
                        _day_site = h.new_site
                    else:
                        if _day_site is None:
                            _day_site = h.old_site  # day predates first transition
                        break
                if _day_site is None:
                    _day_site = employee.site
                data['day_site_id'] = _day_site.id if _day_site else None
                data['day_site_name'] = _day_site.name if _day_site else None
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
                # on that day from EmployeeSiteHistory.
                #
                # Tie-breakers:
                #   - day inside a recorded segment → that segment's new_site
                #   - day BEFORE the earliest segment → the earliest segment's
                #     `old_site` (i.e., what the employee was at before the first
                #     recorded transition). This is what you want when admins
                #     re-assign an employee today: past calendar days should keep
                #     showing the previous site, not the new one.
                #   - no history at all → fall back to current employee.site
                site_segments = list(
                    employee.site_history
                            .select_related('new_site', 'old_site')
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
                    # Day predates the earliest recorded transition —
                    # use that transition's old_site, not the current site.
                    if site_segments:
                        first = site_segments[0]
                        return first.old_site.name if first.old_site else None
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
            
            # If there's no KML polygon, fall back to the circular geofence
            # (center + radius) so the map can still show the boundary.
            if not map_coords and site.geofence_lat is not None and site.geofence_lng is not None:
                center_lat = site.geofence_lat
                center_lng = site.geofence_lng

            return Response({
                'success': True,
                'coordinates': map_coords,
                'center': {
                    'lat': center_lat,
                    'lng': center_lng
                },
                'has_coordinates': len(map_coords) > 0,
                # Circular-geofence fallback (used when there's no KML polygon)
                'geofence_lat': site.geofence_lat,
                'geofence_lng': site.geofence_lng,
                'geofence_radius_meters': site.geofence_radius_meters,
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
                'role': admin.role,
                'role_display': dict(AdminProfile.ROLE_CHOICES).get(admin.role, admin.role),
                'site_id': first_site.id if first_site else '', # Kept for basic compat
                'site_ids': list(assigned_sites.values_list('id', flat=True)),
                'site_name': (
                    'All sites' if admin.role == AdminProfile.ROLE_VIEWER
                    else (", ".join([s.name for s in assigned_sites]) if assigned_sites.exists() else 'No Site')
                ),
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
    role = request.POST.get("role") or AdminProfile.ROLE_SITE_ADMIN
    if role not in (AdminProfile.ROLE_SITE_ADMIN, AdminProfile.ROLE_VIEWER):
        role = AdminProfile.ROLE_SITE_ADMIN
    site_ids = request.POST.getlist("sites")

    # Viewers see ALL sites, so they don't pick any. Site admins must.
    if not (username and email and password):
        return redirect("admin-site-admins")
    if role == AdminProfile.ROLE_SITE_ADMIN and not site_ids:
        return redirect("admin-site-admins")

    try:
        with transaction.atomic():
            if User.objects.filter(username=username).exists():
                return redirect("admin-site-admins")

            user = User.objects.create_user(username=username, email=email, password=password)
            user.is_staff = True
            user.save()

            profile = AdminProfile.objects.create(user=user, role=role)
            if role == AdminProfile.ROLE_VIEWER:
                profile.sites.set(Site.objects.all())   # all sites → sees everything
            else:
                profile.sites.set(Site.objects.filter(id__in=site_ids))

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
    role = request.POST.get("role") or AdminProfile.ROLE_SITE_ADMIN
    if role not in (AdminProfile.ROLE_SITE_ADMIN, AdminProfile.ROLE_VIEWER):
        role = AdminProfile.ROLE_SITE_ADMIN

    try:
        with transaction.atomic():
            user.username = username
            user.email = email
            if password:
                user.set_password(password)
            user.save()

            # Update or create profile + role. Viewers get all sites.
            profile, created = AdminProfile.objects.get_or_create(user=user)
            profile.role = role
            profile.save()
            if role == AdminProfile.ROLE_VIEWER:
                profile.sites.set(Site.objects.all())
            else:
                profile.sites.set(Site.objects.filter(id__in=site_ids))

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


# ---------- Shared exhaustive-export helper ----------

# Every Employee field that should land in the export, in column order.
# Header label  →  (callable extracting the value from an Employee instance)
EMPLOYEE_EXPORT_COLUMNS = [
    # Identity
    ('Employee ID',          lambda e: e.id),
    ('Badge ID',             lambda e: e.badge_number or '-'),
    ('Full Name',            lambda e: e.name or '-'),
    ('Email',                lambda e: e.email or '-'),
    ('Phone',                lambda e: e.phone or '-'),
    # Org placement
    ('Site',                 lambda e: e.site.name if e.site else '-'),
    ('Department',           lambda e: e.department or '-'),
    ('Position',             lambda e: e.position or '-'),
    ('Category (Staff/Worker)', lambda e: e.category.capitalize() if e.category else '-'),
    ('Salary Grade',         lambda e: e.salary_grade or '-'),
    ('Sponsor',              lambda e: e.sponsor or '-'),
    ('Employer',             lambda e: e.employer or '-'),
    ('Job Description',      lambda e: e.job_description or '-'),
    # Personal
    ('Date of Birth',        lambda e: str(e.date_of_birth) if e.date_of_birth else '-'),
    ('Date of Joining',      lambda e: str(e.date_of_joining) if e.date_of_joining else '-'),
    ('Nationality',          lambda e: e.nationality or '-'),
    ('Gender',               lambda e: e.gender or '-'),
    ('Marital Status',       lambda e: e.marital_status or '-'),
    ('Religion',             lambda e: e.religion or '-'),
    # Documents
    ('Passport Number',      lambda e: e.passport_number or '-'),
    ('Passport Expiry',      lambda e: str(e.passport_expiry) if e.passport_expiry else '-'),
    ('Visa Details',         lambda e: e.visa_details or '-'),
    ('Visa Expiry',          lambda e: str(e.visa_expiry_date) if e.visa_expiry_date else '-'),
    ('Labour Card / CEC Nr', lambda e: e.labor_card_number or '-'),
    ('MOL ID',               lambda e: e.mol_id or '-'),
    # Housing
    ('Housing Camp',         lambda e: e.camp or '-'),
    ('Working Type',         lambda e: e.working_type or '-'),
    ('Working Shift',        lambda e: e.working_shift or '-'),
    ('Transportation',       lambda e: e.transportation or '-'),
    # Status + termination
    ('Status',               lambda e: e.status or '-'),
    ('Resumption Date',      lambda e: str(e.resumption_date) if e.resumption_date else '-'),
    ('Last Working Date',    lambda e: str(e.last_working_date) if e.last_working_date else '-'),
    ('Termination Reason',   lambda e: e.termination_reason or '-'),
    # Leave
    ('Emirates ID',          lambda e: e.emirates_id or '-'),
    ('Leave Type',           lambda e: e.leave_type or '-'),
    ('Leave Start',          lambda e: str(e.leave_start_date) if e.leave_start_date else '-'),
    ('Leave End',            lambda e: str(e.leave_end_date) if e.leave_end_date else '-'),
    ('Leave Approval Date',  lambda e: str(e.leave_approval_date) if e.leave_approval_date else '-'),
    ('Leave Ticket Eligible',lambda e: 'Yes' if e.leave_ticket_eligible is True else ('No' if e.leave_ticket_eligible is False else '-')),
    ('Leave Ticket Price',   lambda e: str(e.leave_ticket_price) if e.leave_ticket_price is not None else '-'),
    # Salary breakdown
    ('Basic Salary',         lambda e: str(e.basic_salary) if e.basic_salary is not None else '-'),
    ('Accommodation Allowance', lambda e: str(e.accommodation_allowance) if e.accommodation_allowance is not None else '-'),
    ('Transport Allowance',  lambda e: str(e.transport_allowance) if e.transport_allowance is not None else '-'),
    ('Food Allowance',       lambda e: str(e.food_allowance) if e.food_allowance is not None else '-'),
    ('Fixed OT Allowance',   lambda e: str(e.fixed_ot_allowance) if e.fixed_ot_allowance is not None else '-'),
    ('Other Allowance',      lambda e: str(e.other_allowance) if e.other_allowance is not None else '-'),
    ('Salary Reduction',     lambda e: str(e.salary_reduction) if e.salary_reduction is not None else '-'),
    ('Gross Salary',         lambda e: str(e.gross_salary) if e.gross_salary is not None else '-'),
    ('Salary Remarks',       lambda e: e.salary_remarks or '-'),
]


def _build_employee_workbook(employees, sheet_title='Employees'):
    """Render an exhaustive .xlsx workbook for the given employee queryset.

    Includes EVERY field on the Employee model — identity, org placement,
    documents (dates included), housing, status, leave, full salary breakdown,
    remarks. Used by both the filtered export and the selected-rows export so
    they stay in lock-step.
    """
    import openpyxl
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_title

    headers = [h for h, _ in EMPLOYEE_EXPORT_COLUMNS]
    ws.append(headers)

    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF')
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for col in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center

    for emp in employees:
        ws.append([extractor(emp) for _, extractor in EMPLOYEE_EXPORT_COLUMNS])

    # Reasonable default column widths
    for col in range(1, len(headers) + 1):
        letter = get_column_letter(col)
        # Make name / remarks / job description wider
        name = headers[col - 1].lower()
        if any(k in name for k in ('name', 'description', 'remarks', 'reason')):
            ws.column_dimensions[letter].width = 32
        else:
            ws.column_dimensions[letter].width = 18
    ws.freeze_panes = 'A2'
    return wb


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

        # Position / Department / Category filters
        position_filter = request.GET.get('position')
        department_filter = request.GET.get('department')
        category_filter = request.GET.get('category')
        if position_filter and position_filter != 'all':
            employees = employees.filter(
                Q(position__iexact=position_filter) |
                (Q(position__in=['', None]) & Q(salary_grade__iexact=position_filter))
            )
        if department_filter and department_filter != 'all':
            employees = employees.filter(department__iexact=department_filter)
        if category_filter and category_filter != 'all':
            employees = employees.filter(category__iexact=category_filter)

        # Search
        search = request.GET.get('search')
        if search:
            employees = employees.filter(
                Q(name__icontains=search) | 
                Q(email__icontains=search) | 
                Q(badge_number__icontains=search)
            )

        # Attendance Filter — keep in lock-step with EmployeeListView
        attendance_filter = request.GET.get('attendance_filter')
        if attendance_filter in ('present', 'late'):
            today = timezone.localdate()
            attendance_qs = Attendance.objects.filter(date=today)
            if attendance_filter == 'late':
                attendance_qs = attendance_qs.filter(status='late')
            else:
                attendance_qs = attendance_qs.filter(status='present')
            employee_ids = attendance_qs.values_list('user_id', flat=True)
            employees = employees.filter(id__in=employee_ids)
        elif attendance_filter == 'absent':
            today = timezone.localdate()
            today_ids = Attendance.objects.filter(date=today).values_list('user_id', flat=True).distinct()
            employees = employees.filter(status__iexact='Active').exclude(id__in=today_ids)

        # Render every field on each Employee (shared with the selected-rows export)
        wb = _build_employee_workbook(employees, sheet_title='Filtered Employees')
        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=employees_filtered_{timezone.localdate()}.xlsx'
        wb.save(response)
        return response


class ExportSelectedEmployeesView(APIView):
    """POST /api/attendance/employees/export-selected/

    Body: form-encoded `ids=1,2,3` (or JSON {"ids": [1,2,3]} / {"ids": "1,2,3"}).
    Returns an .xlsx with the same column layout as the filtered export, but
    limited to the explicitly-selected rows the admin checked on the dashboard.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def post(self, request):
        import openpyxl
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, PatternFill
        from django.http import HttpResponse

        # Parse the id list (form OR json, comma-separated string OR list)
        raw = request.data.get('ids')
        if isinstance(raw, str):
            id_list = [s for s in raw.split(',') if s.strip()]
        elif isinstance(raw, (list, tuple)):
            id_list = list(raw)
        else:
            id_list = []
        ids = []
        for v in id_list:
            try:
                ids.append(int(v))
            except (TypeError, ValueError):
                continue
        if not ids:
            return HttpResponse('Pick at least one employee.', status=400)

        employees = (Employee.objects
                     .select_related('site')
                     .filter(id__in=ids)
                     .order_by('name'))
        # Scope site admins to their own sites.
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                employees = employees.filter(site__in=profile.sites.all())
            except AdminProfile.DoesNotExist:
                employees = employees.none()

        # Same exhaustive every-field layout as the filtered export.
        wb = _build_employee_workbook(employees, sheet_title='Selected Employees')
        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=employees_selected_{timezone.localdate()}.xlsx'
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

        # Employment-status quick-filter (same param the page uses)
        emp_status_filter = request.GET.get('emp_status', 'all')
        if emp_status_filter and emp_status_filter != 'all':
            _MASTER = ['Active', 'Leave', 'Resigned', 'Terminated', 'No Renewal', 'Absconding']
            if emp_status_filter == 'Other':
                employees = employees.exclude(status__in=_MASTER)
            else:
                employees = employees.filter(status__iexact=emp_status_filter)

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

REPORT_TERMINAL_STATUSES = {'Resigned', 'Terminated', 'No Renewal', 'Absconding'}


def employment_status_on(emp, on_date):
    """What an employee's state was on `on_date` when there is NO attendance row.

    Without this every non-punch reads as "Absent" — so people on approved leave
    or who already left the company show up as absent in reports. Honours the
    leave window, resumption date and last working date.
    """
    emp_status = (emp.status or '').strip()

    # Already left the company before this date → report the leaving status.
    if emp_status in REPORT_TERMINAL_STATUSES:
        if emp.last_working_date and emp.last_working_date < on_date:
            return emp_status
        # Still employed on this date (last working day not yet reached) → absent.
        if not emp.last_working_date:
            return emp_status

    # Inside an approved leave window.
    if emp.leave_start_date and emp.leave_end_date:
        if emp.leave_start_date <= on_date <= emp.leave_end_date:
            return 'Leave'

    # Flagged as on Leave and not yet resumed by this date.
    if emp_status == 'Leave':
        if not emp.resumption_date or on_date < emp.resumption_date:
            return 'Leave'

    return 'Absent'


def site_map_on_date(employees, on_date):
    """{employee_id: Site} — the site each employee was assigned to on `on_date`.

    Walks EmployeeSiteHistory by `effective_from` so a report shows the site the
    person actually worked at that day, not wherever they were transferred to
    later. All history is fetched in ONE query, so this is safe for thousands of
    employees. Falls back to the employee's current site when there's no history.
    """
    from collections import defaultdict
    hist = defaultdict(list)
    qs = (EmployeeSiteHistory.objects
          .filter(employee__in=employees)
          .select_related('new_site', 'old_site')
          .order_by('effective_from', 'id'))
    for h in qs:
        hist[h.employee_id].append(h)

    out = {}
    for emp in employees:
        day_site = None
        for h in hist.get(emp.id, []):
            if h.effective_from <= on_date:
                day_site = h.new_site
            else:
                if day_site is None:
                    day_site = h.old_site   # date predates the first transfer
                break
        out[emp.id] = day_site or emp.site
    return out


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
    department_filter = request.GET.get('department')
    category_filter = request.GET.get('category')  # Staff / Worker enum
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
        employees = employees.filter(
            Q(position__iexact=position_filter) |
            (Q(position__in=['', None]) & Q(salary_grade__iexact=position_filter))
        )
    if department_filter and department_filter != 'all':
        employees = employees.filter(department__iexact=department_filter)
    if category_filter and category_filter != 'all':
        employees = employees.filter(category__iexact=category_filter)

    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)

    # ── Only people actually employed on the selected date ───────────────────
    # Without this, anyone who has left (Resigned / Terminated / No Renewal /
    # Absconding) still lands in the report and — having no attendance row —
    # is counted as "Absent", inflating the absent totals.
    # Someone who left is still included for dates up to their last working day,
    # so historical reports stay accurate. Pass ?include_inactive=1 to override.
    if (request.GET.get('include_inactive') or '').strip().lower() not in ('1', 'true', 'yes'):
        TERMINAL_STATUSES = ['Resigned', 'Terminated', 'No Renewal', 'Absconding']
        employees = employees.exclude(
            Q(status__in=TERMINAL_STATUSES) & (
                Q(last_working_date__isnull=True) | Q(last_working_date__lt=selected_date)
            )
        )
        # Not yet joined on that date → shouldn't appear either.
        employees = employees.exclude(date_of_joining__gt=selected_date)

    # Materialise once — we iterate the list several times below.
    employees = list(employees.select_related('site'))
    # Site the employee actually belonged to on this date (honours transfers).
    site_on_date = site_map_on_date(employees, selected_date)

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
        check_in = '-'
        check_out = '-'
        _day_site = site_on_date.get(emp.id) or emp.site
        site_name = _day_site.name if _day_site else '-'
        position_name = emp.position or '-'

        if att:
            # A real punch wins, unless the day was explicitly marked sick/leave.
            att_status = (att.status or '').strip().lower()
            if att_status == 'sick':
                status = 'Sick'
            elif att_status == 'leave':
                status = 'Leave'
            else:
                status = 'Present'
            check_in = timezone.localtime(att.check_in_time).strftime('%I:%M %p') if att.check_in_time else '-'
            check_out = timezone.localtime(att.check_out_time).strftime('%I:%M %p') if att.check_out_time else '-'
        else:
            # No punch → reflect the real employment state on that date
            # (Leave / Resigned / Terminated / No Renewal) instead of "Absent".
            status = employment_status_on(emp, selected_date)

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
                emp.working_type or '-',
                emp.working_shift or '-',
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
        
        # Only a genuine no-show counts as Absent. Leave / Sick / leavers are
        # legitimate non-attendance and must not inflate the absent totals.
        if status == 'Present':
            summary_map[key]['present'] += 1
        elif status == 'Absent':
            summary_map[key]['absent'] += 1

    # Create XLSX
    wb = openpyxl.Workbook()
    
    # --- Sheet 1: Detailed Data ---
    ws_detailed = wb.active
    ws_detailed.title = "Detailed Attendance"
    
    headers = ['Employee Name', 'Badge ID', 'Grade', 'Department', 'Position', 'Site', 'Housing Camp', 'Transportation', 'Working Type', 'Working Shift', 'Date', 'Status', 'Check In', 'Check Out']
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

# ────────────────────────────────────────────────────────────────────────────
# Monthly attendance report — analytics helpers
# ────────────────────────────────────────────────────────────────────────────

def _month_dates(year, month):
    n = calendar.monthrange(year, month)[1]
    return [datetime(year, month, d).date() for d in range(1, n + 1)]


def _prev_month(year, month):
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _attendance_band(pct):
    if pct >= 95: return 'champion'
    if pct >= 85: return 'steady'
    if pct >= 70: return 'at_risk'
    return 'critical'


def _working_days_for(emp, dates_in_month, num_days):
    """Working days in the month for a given employee, honouring the weekly
    day-off on their site (different default for staff vs worker)."""
    if not emp.site:
        return num_days
    is_staff = (emp.category or '').lower() == 'staff'
    raw_off = emp.site.office_day_off if is_staff else emp.site.worker_day_off
    day_off = (raw_off or '').strip().lower()
    if not day_off:
        return num_days
    wd = sum(1 for d in dates_in_month
             if d.strftime('%A').lower() != day_off)
    return wd or num_days


def _build_monthly_analytics(employees, attendance_records, year, month):
    """Single-pass analytics builder used by both the JSON endpoint and the
    AI executive-summary endpoint."""

    num_days = calendar.monthrange(year, month)[1]
    dates_in_month = _month_dates(year, month)
    emp_count = len(employees)

    emp_site = {e.id: e.site_id for e in employees}
    emp_dept = {e.id: ((e.department or 'Unassigned').strip() or 'Unassigned')
                for e in employees}
    site_name = {}
    site_emps = defaultdict(list)
    dept_emps = defaultdict(list)
    for e in employees:
        if e.site_id:
            site_name[e.site_id] = e.site.name if e.site else '-'
            site_emps[e.site_id].append(e)
        dept_emps[emp_dept[e.id]].append(e)

    rec_by_user = defaultdict(list)
    trend_p = defaultdict(int)
    trend_l = defaultdict(int)
    site_p = defaultdict(lambda: defaultdict(int))
    dept_p = defaultdict(int)
    dept_l = defaultdict(int)

    for r in attendance_records:
        rec_by_user[r.user_id].append(r)
        if r.check_in_time:
            trend_p[r.date] += 1
            sid = emp_site.get(r.user_id)
            if sid:
                site_p[sid][r.date] += 1
            dept_p[emp_dept.get(r.user_id, 'Unassigned')] += 1
        if r.late_minutes and r.late_minutes > 0:
            trend_l[r.date] += 1
            dept_l[emp_dept.get(r.user_id, 'Unassigned')] += 1

    employee_rows = []
    bands = {'champion': 0, 'steady': 0, 'at_risk': 0, 'critical': 0}
    total_present = 0
    total_late = 0
    total_ot = 0.0
    zero_attendees = []

    for e in employees:
        recs = rec_by_user.get(e.id, [])
        days_present = sum(1 for r in recs if r.check_in_time)
        late_count = sum(1 for r in recs if r.late_minutes and r.late_minutes > 0)
        ot_hours = sum(float(r.normal_ot_hours or 0) + float(r.special_ot_hours or 0)
                       for r in recs)

        working_days = _working_days_for(e, dates_in_month, num_days)
        raw_pct = (days_present / num_days * 100) if num_days else 0
        eff_pct = min((days_present / working_days * 100) if working_days else 0, 100.0)

        bands[_attendance_band(eff_pct)] += 1
        if days_present == 0:
            zero_attendees.append(e)

        total_present += days_present
        total_late += late_count
        total_ot += ot_hours

        employee_rows.append({
            'id': e.id,
            'name': e.name,
            'badge_number': e.badge_number,
            'department': e.department or '-',
            'profile_picture': e.profile_picture.url if e.profile_picture else None,
            'site_id': e.site_id,
            'site': e.site.name if e.site else '-',
            'category': e.category or '',
            'days_present': days_present,
            'days_absent': num_days - days_present,
            'working_days': working_days,
            'late_count': late_count,
            'overtime_hours': round(ot_hours, 2),
            'attendance_percentage': round(eff_pct, 2),
            'attendance_percentage_raw': round(raw_pct, 2),
        })

    trend = []
    for d in dates_in_month:
        p = trend_p.get(d, 0)
        l = trend_l.get(d, 0)
        trend.append({
            'date': d.isoformat(),
            'day': d.day,
            'weekday': d.strftime('%a'),
            'present': p,
            'absent': max(0, emp_count - p),
            'late': l,
        })

    wk_buckets = defaultdict(lambda: {'present': 0, 'late': 0, 'days': 0})
    for t in trend:
        b = wk_buckets[t['weekday']]
        b['present'] += t['present']
        b['late'] += t['late']
        b['days'] += 1
    weekday_split = []
    for wd in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        b = wk_buckets.get(wd, {'present': 0, 'late': 0, 'days': 0})
        avg_pct = (b['present'] / b['days'] / emp_count * 100) if (b['days'] and emp_count) else 0
        weekday_split.append({
            'weekday': wd,
            'avg_present': round(b['present'] / b['days'] if b['days'] else 0, 1),
            'avg_present_pct': round(avg_pct, 1),
            'total_late': b['late'],
            'days_in_month': b['days'],
        })

    heatmap = []
    site_avgs = []
    for sid, emps in site_emps.items():
        cells = []
        site_total_present = 0
        for d in dates_in_month:
            p = site_p[sid].get(d, 0)
            pct = (p / len(emps) * 100) if emps else 0
            cells.append({'day': d.day, 'present': p, 'pct': round(pct, 1)})
            site_total_present += p
        avg_pct = (site_total_present / (len(emps) * num_days) * 100) if (emps and num_days) else 0
        heatmap.append({
            'site_id': sid,
            'site': site_name.get(sid, '-'),
            'employee_count': len(emps),
            'days': cells,
            'avg_pct': round(avg_pct, 1),
        })
        site_avgs.append({
            'site_id': sid,
            'site': site_name.get(sid, '-'),
            'avg_pct': round(avg_pct, 1),
            'employee_count': len(emps),
        })
    heatmap.sort(key=lambda r: -r['avg_pct'])
    site_avgs.sort(key=lambda r: -r['avg_pct'])

    department_breakdown = []
    for dept, emps in dept_emps.items():
        d_p = dept_p.get(dept, 0)
        avg = (d_p / (len(emps) * num_days) * 100) if (emps and num_days) else 0
        department_breakdown.append({
            'department': dept,
            'employee_count': len(emps),
            'avg_pct': round(avg, 1),
            'total_late': dept_l.get(dept, 0),
        })
    department_breakdown.sort(key=lambda r: -r['avg_pct'])

    sorted_by_pct = sorted(employee_rows, key=lambda r: -r['attendance_percentage'])
    top10 = sorted_by_pct[:10]
    bottom_eligible = [r for r in sorted_by_pct if r['days_present'] > 0]
    bottom10 = sorted(bottom_eligible, key=lambda r: r['attendance_percentage'])[:10]

    anomalies = []
    for row in heatmap:
        streak = 0
        max_streak = 0
        for d in row['days']:
            if d['present'] == 0:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
        if max_streak >= 3:
            anomalies.append({
                'severity': 'warning',
                'icon': 'ri-error-warning-line',
                'title': 'Inactive site',
                'message': f"{row['site']} had 0 check-ins for {max_streak} consecutive days",
            })

    if zero_attendees:
        anomalies.append({
            'severity': 'danger',
            'icon': 'ri-user-unfollow-line',
            'title': 'Possible ghosts',
            'message': f"{len(zero_attendees)} employees logged 0 attendance this month — review for offboarding",
        })

    nz_wk = [w for w in weekday_split if w['days_in_month'] > 0]
    avg_late_per_wk = sum(w['total_late'] for w in nz_wk) / len(nz_wk) if nz_wk else 0
    if avg_late_per_wk > 0:
        for w in nz_wk:
            if w['total_late'] > 2 * avg_late_per_wk and w['total_late'] >= 30:
                ratio = int(w['total_late'] / avg_late_per_wk * 100)
                anomalies.append({
                    'severity': 'info',
                    'icon': 'ri-time-line',
                    'title': 'Late spike',
                    'message': f"Lates spike on {w['weekday']}: {w['total_late']} late arrivals ({ratio}% of weekday avg)",
                })

    if emp_count and bands['critical'] / emp_count * 100 >= 10 and bands['critical'] >= 5:
        anomalies.append({
            'severity': 'danger',
            'icon': 'ri-alarm-warning-line',
            'title': 'Critical band high',
            'message': f"{bands['critical']} employees ({bands['critical']/emp_count*100:.0f}%) are below 70% attendance",
        })

    return {
        'rows': employee_rows,
        'trend': trend,
        'heatmap': heatmap,
        'weekday_split': weekday_split,
        'department_breakdown': department_breakdown,
        'site_leaderboard': {
            'top': site_avgs[:5],
            'bottom': list(reversed(site_avgs[-5:])) if len(site_avgs) >= 5 else list(reversed(site_avgs)),
        },
        'employee_leaderboard': {
            'top': top10,
            'bottom': bottom10,
        },
        'bands': bands,
        'anomalies': anomalies,
        'totals': {
            'present': total_present,
            'late': total_late,
            'overtime_hours': round(total_ot, 1),
            'employees_with_zero_attendance': len(zero_attendees),
        },
    }


def _prev_month_summary(employees_qs, year, month):
    """Light comparison summary for the previous month — totals only, no
    per-employee or per-site detail. Used for MoM deltas on the KPI tiles."""
    pm_year, pm_month = _prev_month(year, month)
    pm_days = calendar.monthrange(pm_year, pm_month)[1]
    pm_start = datetime(pm_year, pm_month, 1).date()
    pm_end = datetime(pm_year, pm_month, pm_days).date()

    qs = Attendance.objects.filter(
        date__gte=pm_start,
        date__lte=pm_end,
        user__in=employees_qs,
    )
    pm_present = qs.filter(check_in_time__isnull=False).count()
    pm_late = qs.filter(late_minutes__gt=0).count()

    agg = qs.aggregate(ot=Sum('normal_ot_hours'), sot=Sum('special_ot_hours'))
    pm_ot = float(agg.get('ot') or 0) + float(agg.get('sot') or 0)

    emp_count = employees_qs.count()
    pm_avg = round(pm_present / (emp_count * pm_days) * 100, 2) if (emp_count and pm_days) else 0

    return {
        'month': calendar.month_name[pm_month],
        'month_num': pm_month,
        'year': pm_year,
        'num_days': pm_days,
        'avg_attendance': pm_avg,
        'total_present': pm_present,
        'total_late': pm_late,
        'overtime_hours': round(pm_ot, 1),
    }


@login_required
def monthly_report_view(request):
    """Monthly attendance report — page render + JSON endpoint.

    JSON response shape:
        summary, comparison, trend, heatmap, weekday_split,
        department_breakdown, site_leaderboard, employee_leaderboard,
        bands, anomalies, results (paginated rows), pagination.
    """
    is_superuser = request.user.is_superuser
    try:
        admin_profile = request.user.admin_profile
        site_admin_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        site_admin_sites = Site.objects.none()

    if not is_superuser and not admin_profile:
        return render(request, 'dashboard.html', {'error': 'Permission Denied'})

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

    if not is_ajax:
        now = datetime.now()
        months = [(i, calendar.month_name[i]) for i in range(1, 13)]
        years = list(range(now.year - 2, now.year + 1))
        sites = list(Site.objects.all().values('id', 'name')) if is_superuser else []
        return render(request, 'monthly_report.html', {
            'months': months,
            'years': years,
            'sites': sites,
            'is_superuser': is_superuser,
            'current_month': now.month,
            'current_year': now.year,
        })

    try:
        month = int(request.GET.get('month', datetime.now().month))
        year = int(request.GET.get('year', datetime.now().year))
    except (TypeError, ValueError):
        month, year = datetime.now().month, datetime.now().year
    site_id = request.GET.get('site')
    employer_filter = request.GET.get('employer')
    search_query = request.GET.get('search', '').strip()
    page_num = request.GET.get('page', 1)
    try:
        per_page = int(request.GET.get('per_page', 20))
    except (TypeError, ValueError):
        per_page = 20

    employees = Employee.objects.select_related('site').all()
    if not is_superuser and site_admin_sites.exists():
        employees = employees.filter(site__in=site_admin_sites)
    if is_superuser and site_id and site_id != 'all':
        try:
            employees = employees.filter(site_id=int(site_id))
        except (ValueError, TypeError):
            pass
    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)

    # ── Only staff employed during this month ────────────────────────────────
    # Otherwise leavers show a full month of "Absent" days in the report.
    if (request.GET.get('include_inactive') or '').strip().lower() not in ('1', 'true', 'yes'):
        _m_start = datetime(year, month, 1).date()
        _m_end = datetime(year, month, calendar.monthrange(year, month)[1]).date()
        TERMINAL_STATUSES = ['Resigned', 'Terminated', 'No Renewal', 'Absconding']
        employees = employees.exclude(
            Q(status__in=TERMINAL_STATUSES) & (
                Q(last_working_date__isnull=True) | Q(last_working_date__lt=_m_start)
            )
        )
        employees = employees.exclude(date_of_joining__gt=_m_end)

    analytics_employees = list(employees)

    if search_query:
        rows_qs = employees.filter(
            Q(name__icontains=search_query) |
            Q(badge_number__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(phone__icontains=search_query)
        )
        row_id_filter = {e.id for e in rows_qs}
    else:
        row_id_filter = None

    num_days = calendar.monthrange(year, month)[1]
    dates_in_month = _month_dates(year, month)
    start_date = dates_in_month[0]
    end_date = dates_in_month[-1]

    attendance_records = list(Attendance.objects.filter(
        date__gte=start_date,
        date__lte=end_date,
        user__in=analytics_employees,
    ))

    analytics = _build_monthly_analytics(
        analytics_employees, attendance_records, year, month
    )

    rows = (analytics['rows'] if row_id_filter is None
            else [r for r in analytics['rows'] if r['id'] in row_id_filter])

    emp_count = len(analytics_employees)
    avg_pct_raw = (analytics['totals']['present'] / (emp_count * num_days) * 100
                   if (emp_count and num_days) else 0)
    if analytics['rows']:
        avg_pct_eff = sum(r['attendance_percentage'] for r in analytics['rows']) / len(analytics['rows'])
    else:
        avg_pct_eff = 0

    comparison = _prev_month_summary(employees, year, month)
    delta_avg = round(avg_pct_raw - comparison['avg_attendance'], 1)
    delta_present = analytics['totals']['present'] - comparison['total_present']
    delta_late = analytics['totals']['late'] - comparison['total_late']
    delta_ot = round(analytics['totals']['overtime_hours'] - comparison['overtime_hours'], 1)

    paginator = Paginator(rows, per_page)
    try:
        page_obj = paginator.page(page_num)
    except (PageNotAnInteger, EmptyPage):
        page_obj = paginator.page(1)

    return JsonResponse({
        'results': list(page_obj),
        'summary': {
            'total_days': num_days,
            'total_employees': emp_count,
            'avg_attendance': round(avg_pct_raw, 2),
            'avg_attendance_effective': round(avg_pct_eff, 2),
            'total_present': analytics['totals']['present'],
            'total_late': analytics['totals']['late'],
            'total_overtime': analytics['totals']['overtime_hours'],
            'employees_with_zero_attendance': analytics['totals']['employees_with_zero_attendance'],
            'month_name': calendar.month_name[month],
            'year': year,
            'month_num': month,
        },
        'comparison': {
            'previous': comparison,
            'delta_avg_pct': delta_avg,
            'delta_present': delta_present,
            'delta_late': delta_late,
            'delta_overtime': delta_ot,
        },
        'trend': analytics['trend'],
        'heatmap': analytics['heatmap'],
        'weekday_split': analytics['weekday_split'],
        'department_breakdown': analytics['department_breakdown'],
        'site_leaderboard': analytics['site_leaderboard'],
        'employee_leaderboard': analytics['employee_leaderboard'],
        'bands': analytics['bands'],
        'anomalies': analytics['anomalies'],
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
            'is_superuser': is_superuser,
        },
    })


@login_required
def monthly_report_employee_calendar(request, employee_id):
    """Returns one employee's daily attendance for a given month in a flat
    list ready for the mini-calendar modal."""
    try:
        emp = Employee.objects.select_related('site').get(id=employee_id)
    except Employee.DoesNotExist:
        return JsonResponse({'error': 'Employee not found'}, status=404)

    if not request.user.is_superuser:
        try:
            profile = AdminProfile.objects.get(user=request.user)
            if profile.sites.exists() and emp.site not in profile.sites.all():
                return JsonResponse({'error': 'Permission denied'}, status=403)
        except AdminProfile.DoesNotExist:
            return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        month = int(request.GET.get('month', datetime.now().month))
        year = int(request.GET.get('year', datetime.now().year))
    except (TypeError, ValueError):
        month, year = datetime.now().month, datetime.now().year

    num_days = calendar.monthrange(year, month)[1]
    start_date = datetime(year, month, 1).date()
    end_date = datetime(year, month, num_days).date()

    recs = Attendance.objects.filter(
        user=emp, date__gte=start_date, date__lte=end_date,
    ).order_by('date')

    rows = []
    for r in recs:
        rows.append({
            'date': r.date.isoformat(),
            'check_in_time': r.check_in_time.isoformat() if r.check_in_time else None,
            'check_out_time': r.check_out_time.isoformat() if r.check_out_time else None,
            'late_minutes': r.late_minutes or 0,
            'overtime_hours': float((r.normal_ot_hours or 0) + (r.special_ot_hours or 0)),
            'status': r.status,
        })

    return JsonResponse({
        'employee': {
            'id': emp.id,
            'name': emp.name,
            'badge_number': emp.badge_number,
            'department': emp.department or '',
            'site': emp.site.name if emp.site else '',
        },
        'month': month,
        'year': year,
        'month_name': calendar.month_name[month],
        'num_days': num_days,
        'attendance': rows,
    })


@login_required
def monthly_report_exec_summary(request):
    """AI-polished executive summary for the selected month. Falls back to
    heuristic bullets when AI is disabled or the call fails."""
    is_superuser = request.user.is_superuser
    try:
        admin_profile = request.user.admin_profile
        site_admin_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        site_admin_sites = Site.objects.none()

    if not is_superuser and not admin_profile:
        return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        month = int(request.GET.get('month', datetime.now().month))
        year = int(request.GET.get('year', datetime.now().year))
    except (TypeError, ValueError):
        month, year = datetime.now().month, datetime.now().year
    site_id = request.GET.get('site')
    employer_filter = request.GET.get('employer')

    employees = Employee.objects.select_related('site').all()
    if not is_superuser and site_admin_sites.exists():
        employees = employees.filter(site__in=site_admin_sites)
    if is_superuser and site_id and site_id != 'all':
        try:
            employees = employees.filter(site_id=int(site_id))
        except (ValueError, TypeError):
            pass
    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)

    emp_list = list(employees)
    dates_in_month = _month_dates(year, month)
    start_date = dates_in_month[0]
    end_date = dates_in_month[-1]
    attendance_records = list(Attendance.objects.filter(
        date__gte=start_date, date__lte=end_date, user__in=emp_list,
    ))

    analytics = _build_monthly_analytics(emp_list, attendance_records, year, month)
    prev = _prev_month_summary(employees, year, month)

    num_days = calendar.monthrange(year, month)[1]
    emp_count = len(emp_list)
    total_present = analytics['totals']['present']
    total_late = analytics['totals']['late']
    overtime = analytics['totals']['overtime_hours']
    bands = analytics['bands']
    avg_pct = round(total_present / (emp_count * num_days) * 100, 2) if (emp_count and num_days) else 0

    delta = round(avg_pct - prev['avg_attendance'], 1)
    direction = 'up' if delta > 0 else ('down' if delta < 0 else 'flat')

    top_dept = analytics['department_breakdown'][0] if analytics['department_breakdown'] else None
    worst_dept = analytics['department_breakdown'][-1] if analytics['department_breakdown'] else None
    worst_site = analytics['site_leaderboard']['bottom'][0] if analytics['site_leaderboard']['bottom'] else None

    bullets = [
        {'icon': 'ri-line-chart-line', 'severity': 'info',
         'text': f"Average attendance was {avg_pct}% — {direction} {abs(delta)}pp vs {prev['month']} {prev['year']} ({prev['avg_attendance']}%)."},
        {'icon': 'ri-group-line', 'severity': 'info',
         'text': f"{emp_count:,} active employees logged {total_present:,} present-days, {total_late:,} late arrivals and {overtime:,.0f} overtime hours."},
        {'icon': 'ri-bar-chart-grouped-line', 'severity': 'info',
         'text': f"Bands: {bands['champion']} champions (≥95%), {bands['steady']} steady (85–95%), {bands['at_risk']} at risk (70–85%), {bands['critical']} critical (<70%)."},
    ]
    if top_dept and worst_dept and top_dept['department'] != worst_dept['department']:
        bullets.append({
            'icon': 'ri-building-line', 'severity': 'info',
            'text': f"Best department: {top_dept['department']} at {top_dept['avg_pct']}%. Bottom: {worst_dept['department']} at {worst_dept['avg_pct']}%.",
        })
    if worst_site:
        bullets.append({
            'icon': 'ri-map-pin-line', 'severity': 'warning',
            'text': f"Site needing attention: {worst_site['site']} at {worst_site['avg_pct']}% avg attendance ({worst_site['employee_count']} employees).",
        })
    for a in analytics['anomalies'][:2]:
        bullets.append({
            'icon': a.get('icon', 'ri-error-warning-line'),
            'severity': a.get('severity', 'warning'),
            'text': a['message'],
        })

    ai_polished = None
    ai_used = False
    try:
        from attendance.services import llm as _llm
        if _llm.ai_is_enabled():
            ai_polished = _llm.polish_summary(bullets)
            ai_used = bool(ai_polished)
    except Exception:
        ai_polished = None

    return JsonResponse({
        'narrative': ai_polished or bullets,
        'ai_used': ai_used,
        'heuristic_fallback': bullets,
        'month_name': calendar.month_name[month],
        'year': year,
    })


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

    # ── Only staff employed during this month ────────────────────────────────
    # Leavers (Resigned / Terminated / No Renewal / Absconding) otherwise appear
    # with a full month of "Absent" days. Anyone whose last working day falls in
    # this month is still included, so past months stay accurate.
    if (request.GET.get('include_inactive') or '').strip().lower() not in ('1', 'true', 'yes'):
        TERMINAL_STATUSES = ['Resigned', 'Terminated', 'No Renewal', 'Absconding']
        employees = employees.exclude(
            Q(status__in=TERMINAL_STATUSES) & (
                Q(last_working_date__isnull=True) | Q(last_working_date__lt=start_date)
            )
        )
        employees = employees.exclude(date_of_joining__gt=end_date)
    
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

def _daily_baseline(emp_qs, selected_date):
    """Cheap baseline counts for delta tiles.

    Returns yesterday's present count + the avg present count across the last
    4 same-weekdays. Heavy queries are aggregate-only — no per-row work."""
    yesterday = selected_date - timedelta(days=1)
    y_present = Attendance.objects.filter(
        date=yesterday, user__in=emp_qs, check_in_time__isnull=False,
    ).count()

    same_wd_dates = []
    for w in range(1, 5):
        d = selected_date - timedelta(days=7 * w)
        same_wd_dates.append(d)
    base_present = Attendance.objects.filter(
        date__in=same_wd_dates, user__in=emp_qs, check_in_time__isnull=False,
    ).count()
    base_avg = int(round(base_present / len(same_wd_dates))) if same_wd_dates else 0

    return {
        'yesterday_present': y_present,
        'same_weekday_avg_present': base_avg,
        'same_weekday_dates': [d.isoformat() for d in same_wd_dates],
    }


def _avg_checkin_time(records):
    """Average check-in time (seconds since midnight, local TZ) across records
    with a check_in_time. Returns None if no records."""
    if not records:
        return None
    total = 0
    n = 0
    for r in records:
        if not r.check_in_time:
            continue
        t = timezone.localtime(r.check_in_time)
        total += t.hour * 3600 + t.minute * 60 + t.second
        n += 1
    return total // n if n else None


def _fmt_seconds(s):
    if s is None:
        return None
    h = s // 3600
    m = (s % 3600) // 60
    ampm = 'AM' if h < 12 else 'PM'
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def _seven_day_baseline_checkin(emp_qs, selected_date):
    """Avg check-in time for these employees over the past 7 days."""
    start = selected_date - timedelta(days=7)
    end = selected_date - timedelta(days=1)
    recs = list(Attendance.objects.filter(
        date__gte=start, date__lte=end, user__in=emp_qs,
        check_in_time__isnull=False,
    ).only('check_in_time'))
    return _avg_checkin_time(recs)


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
        department_filter = request.GET.get('department')
        category_filter = request.GET.get('category')
        employer_filter = request.GET.get('employer')
        quick_filter = (request.GET.get('quick') or '').lower()
        page_num = request.GET.get('page', 1)
        try:
            per_page = int(request.GET.get('per_page', 20))
        except (TypeError, ValueError):
            per_page = 20

        # Date logic
        if date_str:
            try:
                selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                selected_date = timezone.localdate()
        else:
            selected_date = timezone.localdate()

        # Build employee queryset
        employees = Employee.objects.select_related('site').all()
        sites_list = []

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

        if position_filter and position_filter != 'all':
            employees = employees.filter(
                Q(position__iexact=position_filter) |
                (Q(position__in=['', None]) & Q(salary_grade__iexact=position_filter))
            )
        if department_filter and department_filter != 'all':
            employees = employees.filter(department__iexact=department_filter)
        if category_filter and category_filter != 'all':
            employees = employees.filter(category__iexact=category_filter)
        if employer_filter and employer_filter != 'all':
            employees = employees.filter(employer__iexact=employer_filter)

        # Available choice lists
        raw_positions = Employee.objects.exclude(position__isnull=True).exclude(position='').values_list('position', flat=True)
        unified_positions = {}
        for p in raw_positions:
            if not p:
                continue
            p_strip = p.strip()
            p_lower = p_strip.lower()
            if p_lower not in unified_positions:
                unified_positions[p_lower] = p_strip.capitalize()
        positions_list = sorted(list(unified_positions.values()))

        # Today's attendance records
        attendance_records = list(Attendance.objects.filter(
            date=selected_date, user__in=employees,
        ).select_related('user', 'user__site'))
        attendance_map = {att.user_id: att for att in attendance_records}

        # Aggregate collections
        all_results = []
        stats = {'total': 0, 'present': 0, 'absent': 0, 'late': 0,
                 'missing_checkout': 0, 'geofence_violations': 0, 'checking_in_now': 0}
        hourly = {h: 0 for h in range(24)}                 # check-in hour -> count
        site_buckets = defaultdict(lambda: {'present': 0, 'total': 0, 'late': 0, 'name': '-'})
        dept_buckets = defaultdict(lambda: {'present': 0, 'total': 0})
        cat_buckets  = defaultdict(lambda: {'present': 0, 'total': 0})

        late_list = []          # employees marked late, sorted by minutes desc
        missing_checkout_list = []
        geofence_violation_list = []

        now_local = timezone.localtime()
        five_min_ago = now_local - timedelta(minutes=5)
        # Schedule lookup for "expected check-out" calc
        eod_threshold = now_local - timedelta(hours=6)  # any check-in older than 6h with no check-out is "missing"

        for emp in employees:
            att = attendance_map.get(emp.id)
            site_name = emp.site.name if emp.site else '-'
            site_key  = emp.site_id or 0
            dept_name = (emp.department or 'Unassigned').strip() or 'Unassigned'
            cat_name  = (emp.category or 'unspecified').strip().lower()

            check_in = '-'
            check_out = '-'
            late_min = 0
            ot_hours = 0.0
            is_geofence_violation = False
            checked_in_at_iso = None
            checked_out_at_iso = None

            # No punch → show the real state (Leave / Resigned / Terminated /
            # No Renewal) rather than a blanket "Absent".
            status = employment_status_on(emp, selected_date)

            if att and att.check_in_time:
                status = 'Present'
                in_local = timezone.localtime(att.check_in_time)
                check_in = in_local.strftime('%I:%M %p')
                checked_in_at_iso = in_local.isoformat()
                hourly[in_local.hour] = hourly.get(in_local.hour, 0) + 1
                if att.check_in_time >= five_min_ago:
                    stats['checking_in_now'] += 1
                if att.check_out_time:
                    check_out = timezone.localtime(att.check_out_time).strftime('%I:%M %p')
                    checked_out_at_iso = timezone.localtime(att.check_out_time).isoformat()
                elif att.check_in_time and att.check_in_time < eod_threshold:
                    # Checked in but never checked out, ≥6h ago
                    stats['missing_checkout'] += 1
                    missing_checkout_list.append({
                        'id': emp.id, 'name': emp.name, 'site': site_name,
                        'check_in': check_in, 'department': dept_name,
                        'badge_number': emp.badge_number,
                        'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                    })
                late_min = att.late_minutes or 0
                if late_min > 0:
                    stats['late'] += 1
                    late_list.append({
                        'id': emp.id, 'name': emp.name, 'site': site_name,
                        'check_in': check_in, 'late_minutes': late_min,
                        'department': dept_name, 'badge_number': emp.badge_number,
                        'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                    })
                ot_hours = float((att.normal_ot_hours or 0) + (att.special_ot_hours or 0))
                if att.is_within_geofence is False:
                    is_geofence_violation = True
                    stats['geofence_violations'] += 1
                    geofence_violation_list.append({
                        'id': emp.id, 'name': emp.name, 'site': site_name,
                        'check_in': check_in,
                        'latitude': att.latitude, 'longitude': att.longitude,
                        'badge_number': emp.badge_number,
                        'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                    })

            stats['total'] += 1
            site_buckets[site_key]['name'] = site_name
            site_buckets[site_key]['total'] += 1
            dept_buckets[dept_name]['total'] += 1
            cat_buckets[cat_name]['total']  += 1
            if status == 'Present':
                stats['present'] += 1
                site_buckets[site_key]['present'] += 1
                dept_buckets[dept_name]['present'] += 1
                cat_buckets[cat_name]['present']  += 1
                if late_min > 0:
                    site_buckets[site_key]['late'] += 1
            else:
                stats['absent'] += 1

            # Quick filters narrow the rows but not the stats
            include_row = True
            if status_filter:
                if status_filter.lower() == 'present' and status != 'Present':
                    include_row = False
                if status_filter.lower() == 'absent' and status != 'Absent':
                    include_row = False
            if quick_filter == 'late' and not (att and att.late_minutes and att.late_minutes > 0):
                include_row = False
            elif quick_filter == 'missing_checkout' and not (att and att.check_in_time and not att.check_out_time and att.check_in_time < eod_threshold):
                include_row = False
            elif quick_filter == 'geofence' and not is_geofence_violation:
                include_row = False
            elif quick_filter == 'now' and not (att and att.check_in_time and att.check_in_time >= five_min_ago):
                include_row = False

            if not include_row:
                continue

            all_results.append({
                'id': emp.id,
                'name': emp.name,
                'email': emp.email,
                'badge_number': emp.badge_number,
                'salary_grade': emp.salary_grade,
                'department': emp.department,
                'position': emp.position,
                'profile_picture': emp.profile_picture.url if emp.profile_picture else None,
                'site': site_name,
                'site_id': emp.site.id if emp.site else None,
                'working_type': emp.working_type or '',
                'working_shift': emp.working_shift or '',
                'status': status,
                'check_in': check_in,
                'check_out': check_out,
                'check_in_iso': checked_in_at_iso,
                'check_out_iso': checked_out_at_iso,
                'late_minutes': late_min,
                'overtime_hours': round(ot_hours, 2),
                'is_geofence_violation': is_geofence_violation,
                'latitude': att.latitude if att else None,
                'longitude': att.longitude if att else None,
            })

        all_results.sort(key=lambda x: (x['status'] != 'Present',
                                       -(x.get('late_minutes') or 0)))
        late_list.sort(key=lambda x: -x['late_minutes'])
        late_list = late_list[:50]
        missing_checkout_list = missing_checkout_list[:50]
        geofence_violation_list = geofence_violation_list[:50]

        # Baseline + deltas
        baseline = _daily_baseline(employees, selected_date)
        delta_yesterday = stats['present'] - baseline['yesterday_present']
        delta_baseline  = stats['present'] - baseline['same_weekday_avg_present']

        # Avg check-in time today vs 7-day
        today_recs = [r for r in attendance_records if r.check_in_time]
        avg_checkin_today_s = _avg_checkin_time(today_recs)
        avg_checkin_base_s  = _seven_day_baseline_checkin(employees, selected_date)
        checkin_delta_min = None
        if avg_checkin_today_s is not None and avg_checkin_base_s is not None:
            checkin_delta_min = (avg_checkin_today_s - avg_checkin_base_s) // 60

        # Site/dept summaries with pct
        def with_pct(b):
            t = b['total'] or 1
            return {**b, 'pct': round(b['present'] / t * 100, 1)}

        site_rows = [
            with_pct({**v, 'site_id': k}) for k, v in site_buckets.items() if v['total'] > 0
        ]
        site_rows.sort(key=lambda r: -r['pct'])

        dept_rows = [
            {'department': k, **with_pct(v)} for k, v in dept_buckets.items() if v['total'] > 0
        ]
        dept_rows.sort(key=lambda r: -r['pct'])
        dept_rows = dept_rows[:12]

        category_rows = [
            {'category': k, **with_pct(v)} for k, v in cat_buckets.items() if v['total'] > 0
        ]

        # Anomaly callouts
        anomalies = []
        silent_sites = [s for s in site_rows if s['present'] == 0 and s['total'] > 2]
        for s in silent_sites[:5]:
            anomalies.append({
                'severity': 'warning', 'icon': 'ri-volume-mute-line', 'title': 'Silent site',
                'message': f"{s['name']} has 0 check-ins today ({s['total']} employees assigned)",
            })
        if stats['geofence_violations'] > 0:
            anomalies.append({
                'severity': 'danger', 'icon': 'ri-map-pin-2-line', 'title': 'Geofence',
                'message': f"{stats['geofence_violations']} check-ins outside their site geofence",
            })
        if stats['missing_checkout'] >= 5:
            anomalies.append({
                'severity': 'warning', 'icon': 'ri-logout-box-line', 'title': 'Missing check-outs',
                'message': f"{stats['missing_checkout']} employees clocked in but never clocked out",
            })
        if checkin_delta_min is not None and checkin_delta_min >= 15:
            anomalies.append({
                'severity': 'info', 'icon': 'ri-time-line', 'title': 'Late start',
                'message': f"Avg check-in is {checkin_delta_min} min later than the 7-day baseline",
            })
        if baseline['same_weekday_avg_present'] > 0:
            ratio = stats['present'] / baseline['same_weekday_avg_present']
            if ratio < 0.7 and selected_date == timezone.localdate():
                anomalies.append({
                    'severity': 'danger', 'icon': 'ri-pulse-line', 'title': 'Below baseline',
                    'message': f"Present count ({stats['present']:,}) is {int((1 - ratio) * 100)}% below the same-weekday average ({baseline['same_weekday_avg_present']:,})",
                })

        # End-of-day projection
        projection = None
        if selected_date == timezone.localdate() and stats['present'] > 0:
            hour_decimal = now_local.hour + now_local.minute / 60
            # Assume 95% of daily check-ins occur before 11 AM; project from 11 AM curve
            pivot = 11.0
            if hour_decimal < pivot:
                ratio = max(0.05, hour_decimal / pivot)
                projected = int(stats['present'] / ratio)
                projection = {
                    'projected_present': projected,
                    'hour_decimal': round(hour_decimal, 1),
                    'pivot_hour': pivot,
                    'method': 'morning-curve',
                }
            else:
                projection = {
                    'projected_present': stats['present'],
                    'hour_decimal': round(hour_decimal, 1),
                    'pivot_hour': pivot,
                    'method': 'past-pivot',
                }

        # Categories list for dropdown
        category_choices = sorted(list({
            c.strip().capitalize(): c.strip().capitalize()
            for c in (list(Employee.objects.exclude(category__isnull=True).exclude(category='').values_list('category', flat=True))
                      + list(Employee.objects.exclude(salary_grade__isnull=True).exclude(salary_grade='').values_list('salary_grade', flat=True)))
            if c and c.strip()
        }.values()))

        # Paginate
        paginator = Paginator(all_results, per_page)
        try:
            page_obj = paginator.page(page_num)
        except (PageNotAnInteger, EmptyPage):
            page_obj = paginator.page(1)

        return Response({
            'results': list(page_obj),
            'stats': stats,
            'comparison': {
                'yesterday_present': baseline['yesterday_present'],
                'delta_yesterday': delta_yesterday,
                'same_weekday_avg_present': baseline['same_weekday_avg_present'],
                'delta_baseline': delta_baseline,
                'avg_checkin_today':    _fmt_seconds(avg_checkin_today_s),
                'avg_checkin_baseline': _fmt_seconds(avg_checkin_base_s),
                'avg_checkin_delta_min': checkin_delta_min,
            },
            'hourly_histogram': [{'hour': h, 'count': hourly.get(h, 0)} for h in range(24)],
            'site_rows': site_rows,
            'dept_rows': dept_rows,
            'category_rows': category_rows,
            'anomalies': anomalies,
            'late_list': late_list,
            'missing_checkout_list': missing_checkout_list,
            'geofence_violation_list': geofence_violation_list,
            'projection': projection,
            'sites': sites_list,
            'positions': positions_list,
            'categories': category_choices,
            'employers': list(EMPLOYER_CHOICES),
            'sponsors': list(SPONSOR_CHOICES),
            'selected_site': site_id,
            'selected_date': selected_date.strftime('%Y-%m-%d'),
            'server_now': now_local.isoformat(),
            'permissions': {'is_superuser': is_superuser},
            'pagination': {
                'current_page': page_obj.number,
                'num_pages': paginator.num_pages,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'total_items': paginator.count,
                'start_index': page_obj.start_index(),
                'end_index': page_obj.end_index(),
            }
        })


@login_required
def daily_report_exec_summary(request):
    """AI-polished executive summary for a single day. Falls back to heuristic
    bullets when AI is disabled or the call fails."""
    is_superuser = request.user.is_superuser
    is_staff = request.user.is_staff
    try:
        admin_profile = request.user.admin_profile
        permission_sites = admin_profile.sites.all()
    except AdminProfile.DoesNotExist:
        admin_profile = None
        permission_sites = Site.objects.none()

    if not is_superuser and not is_staff and not admin_profile:
        return JsonResponse({'error': 'Permission denied'}, status=403)

    date_str = request.GET.get('date')
    site_id  = request.GET.get('site')
    employer_filter = request.GET.get('employer')

    if date_str:
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            selected_date = timezone.localdate()
    else:
        selected_date = timezone.localdate()

    employees = Employee.objects.select_related('site').all()
    if not (is_superuser or is_staff) and permission_sites.exists():
        employees = employees.filter(site__in=permission_sites)
    if (is_superuser or is_staff) and site_id and site_id != 'all':
        try:
            employees = employees.filter(site_id=int(site_id))
        except (ValueError, TypeError):
            pass
    if employer_filter and employer_filter != 'all':
        employees = employees.filter(employer__iexact=employer_filter)

    total = employees.count()
    today_recs = list(Attendance.objects.filter(
        date=selected_date, user__in=employees,
    ).select_related('user', 'user__site'))
    present = sum(1 for r in today_recs if r.check_in_time)
    late    = sum(1 for r in today_recs if r.late_minutes and r.late_minutes > 0)
    missing_checkout = sum(1 for r in today_recs
                           if r.check_in_time and not r.check_out_time)
    geofence_violations = sum(1 for r in today_recs if r.is_within_geofence is False)

    baseline = _daily_baseline(employees, selected_date)
    avg_today = _avg_checkin_time([r for r in today_recs if r.check_in_time])
    avg_base  = _seven_day_baseline_checkin(employees, selected_date)
    delta_min = (avg_today - avg_base) // 60 if (avg_today is not None and avg_base is not None) else None

    pct = round(present / total * 100, 1) if total else 0
    delta_y  = present - baseline['yesterday_present']
    delta_b  = present - baseline['same_weekday_avg_present']
    direction_y = 'up' if delta_y > 0 else ('down' if delta_y < 0 else 'flat')

    # Silent sites
    silent = 0
    site_present = defaultdict(int)
    site_total   = defaultdict(int)
    for e in employees:
        site_total[e.site_id] += 1
    for r in today_recs:
        if r.check_in_time:
            site_present[r.user.site_id if r.user else None] += 1
    for sid, t in site_total.items():
        if sid and t > 2 and site_present.get(sid, 0) == 0:
            silent += 1

    bullets = [
        {'icon': 'ri-line-chart-line', 'severity': 'info',
         'text': f"{present:,} of {total:,} present ({pct}%) on {selected_date.strftime('%a, %b %d')} — {direction_y} {abs(delta_y):,} vs yesterday ({baseline['yesterday_present']:,})."},
        {'icon': 'ri-bar-chart-line', 'severity': 'info',
         'text': f"Same-weekday baseline (4-week avg): {baseline['same_weekday_avg_present']:,} present. Today is {'+' if delta_b>=0 else ''}{delta_b:,} vs baseline."},
        {'icon': 'ri-alarm-warning-line', 'severity': 'info',
         'text': f"{late} late arrivals · {missing_checkout} missing check-outs · {geofence_violations} geofence violations."},
    ]
    if delta_min is not None:
        bullets.append({
            'icon': 'ri-time-line',
            'severity': 'warning' if delta_min >= 15 else 'info',
            'text': f"Avg check-in: {_fmt_seconds(avg_today)} — {abs(delta_min):.0f} min {'later' if delta_min > 0 else 'earlier'} than the 7-day baseline ({_fmt_seconds(avg_base)}).",
        })
    if silent:
        bullets.append({
            'icon': 'ri-volume-mute-line', 'severity': 'warning',
            'text': f"{silent} site(s) had 0 check-ins today — investigate whether the site is closed, on holiday, or has an enrolment issue.",
        })

    ai_polished = None
    ai_used = False
    try:
        from attendance.services import llm as _llm
        if _llm.ai_is_enabled():
            ai_polished = _llm.polish_summary(bullets)
            ai_used = bool(ai_polished)
    except Exception:
        ai_polished = None

    return JsonResponse({
        'narrative': ai_polished or bullets,
        'ai_used': ai_used,
        'heuristic_fallback': bullets,
        'date': selected_date.isoformat(),
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


# ---------------------------------------------------------------------------
# Job Categories + Distribution List
# ---------------------------------------------------------------------------

class JobCategoryListView(APIView):
    """List job categories. Used by searchable category dropdowns.

    Query params:
        q    — substring (case-insensitive) match against name / department
        type — 'staff' | 'worker' | 'resource' | omit for all
        limit — default 200, max 2000
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        qs = JobCategory.objects.filter(is_active=True)
        q = (request.GET.get('q') or '').strip()
        t = (request.GET.get('type') or '').strip().lower()
        if t in ('staff', 'worker', 'resource'):
            qs = qs.filter(employee_type=t)
        if q:
            qs = qs.filter(Q(name__icontains=q) | Q(department__icontains=q))
        try:
            limit = max(1, min(int(request.GET.get('limit') or 200), 2000))
        except ValueError:
            limit = 200
        qs = qs[:limit]
        items = [{
            'id': c.id,
            'name': c.name,
            'department': c.department or '',
            'department_id': c.department_fk_id,
            'employee_type': c.employee_type,
        } for c in qs]
        return Response({'count': len(items), 'results': items})


class DepartmentListView(APIView):
    """Distinct department list for the searchable Division/Department dropdown.

    Primary source: active rows from the managed `Department` table (the new
    Department-management page is the source of truth). Legacy fall-back:
    free-text `Employee.department` values that aren't covered by a managed row.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        managed = list(
            Department.objects
            .filter(is_active=True)
            .order_by('sheet_order', 'name')
            .values('id', 'name', 'manager_name')
        )
        managed_keys = {(d['name'] or '').strip().lower() for d in managed}

        legacy_extras = []
        for raw in (Employee.objects
                    .exclude(department__isnull=True).exclude(department='')
                    .values_list('department', flat=True).distinct()):
            d = (raw or '').strip()
            if d and d.lower() not in managed_keys:
                legacy_extras.append({'id': None, 'name': d, 'manager_name': ''})
                managed_keys.add(d.lower())

        items = managed + sorted(legacy_extras, key=lambda x: x['name'].lower())

        q = (request.GET.get('q') or '').strip().lower()
        if q:
            items = [d for d in items if q in (d['name'] or '').lower()]

        return Response({
            'count': len(items),
            'results': [
                {'id': d.get('id'),
                 'name': d['name'],
                 'manager_name': d.get('manager_name') or ''}
                for d in items
            ],
        })


def _distribution_employee_qs(request):
    """Apply site-admin scope to the employee queryset used by the
    distribution endpoints, so site admins only count their own sites."""
    qs = Employee.objects.select_related('site').all()
    if not request.user.is_superuser:
        try:
            profile = AdminProfile.objects.get(user=request.user)
            qs = qs.filter(site__in=profile.sites.all())
        except AdminProfile.DoesNotExist:
            qs = qs.none()
    return qs


def _distribution_payload(request, employee_type):
    """Build the distribution table for a single tab.

    Fully data-driven — both departments and positions come directly from the
    Employee table (legacy salary_grade is honoured as a fall-back for the
    position field). The seeded JobCategory rows only drive the dropdown
    options in the modal; they do NOT decide the row structure here.

    Tabs:
        staff    — Employee.category == 'staff',  site × position matrix
        worker   — Employee.category == 'worker', site × position matrix
        resource — all employees, no site breakdown (flat total per row)

    Payload shapes:

        Staff / Worker:
          { employee_type, sites: [...], rows: [
              {kind:'dept', name:...},
              {kind:'trade', counts: {site_id: n}, leave, total},
              {kind:'subtotal', counts: {...}, leave, total} ], grand_total }

        Resource:
          { employee_type:'resource', sites: [],
            rows: [ {kind:'dept', name:...},
                    {kind:'trade', count: N},
                    {kind:'subtotal', count: N} ], grand_total }
    """
    from collections import defaultdict

    # ---------- Build the employee queryset for this tab ----------
    emps_qs = _distribution_employee_qs(request)
    if employee_type in ('staff', 'worker'):
        emps_qs = emps_qs.filter(category__iexact=employee_type)
    # 'resource' tab = all employees (regardless of category).

    # ---------- Site columns (only used for staff/worker) ----------
    site_id_filter = request.GET.get('site')
    sites_qs = Site.objects.all().order_by('name')
    if site_id_filter and site_id_filter != 'all':
        try:
            sites_qs = sites_qs.filter(id=int(site_id_filter))
        except (ValueError, TypeError):
            pass
    if not request.user.is_superuser:
        try:
            profile = AdminProfile.objects.get(user=request.user)
            sites_qs = sites_qs.filter(id__in=profile.sites.values_list('id', flat=True))
        except AdminProfile.DoesNotExist:
            sites_qs = sites_qs.none()
    site_list = list(sites_qs.values('id', 'name'))
    site_ids = [s['id'] for s in site_list]

    # ---------- Resolve each employee's (department, position) ----------
    # Department / Position both come straight from the Employee row.
    # Legacy fall-back: if Position is blank, use salary_grade (the field that
    # used to back the old "Category" textbox).
    def _resolve(e):
        dept = (e.get('department') or '').strip() or '—'
        pos = (e.get('position') or '').strip()
        if not pos:
            pos = (e.get('salary_grade') or '').strip()
        return dept, (pos or '—')

    # ---------- RESOURCE: flat (department → position → count) ----------
    if employee_type == 'resource':
        counts = defaultdict(int)            # (dept, pos) -> int
        dept_seen_first = {}                  # dept -> first-insert index, for stable ordering
        pos_seen_first = {}                   # (dept, pos) -> first-insert index
        for i, e in enumerate(emps_qs.values('department', 'position', 'salary_grade')):
            d, p = _resolve(e)
            counts[(d, p)] += 1
            dept_seen_first.setdefault(d, i)
            pos_seen_first.setdefault((d, p), i)

        # Order departments alphabetically, positions alphabetically within each.
        depts = sorted(dept_seen_first.keys(), key=lambda x: x.lower())
        rows = []
        grand_total = 0
        for dept in depts:
            rows.append({'kind': 'dept', 'name': dept})
            dept_total = 0
            positions = sorted(
                [(d, p) for (d, p) in counts.keys() if d == dept],
                key=lambda x: x[1].lower(),
            )
            for (d, p) in positions:
                n = counts[(d, p)]
                dept_total += n
                grand_total += n
                rows.append({
                    'kind': 'trade',
                    'department': d,
                    'name': p,
                    'count': n,
                })
            rows.append({'kind': 'subtotal', 'department': dept, 'count': dept_total})

        return {
            'employee_type': 'resource',
            'sites': [],
            'rows': rows,
            'grand_total': grand_total,
            'selected_site': site_id_filter or 'all',
        }

    # ---------- STAFF / WORKER: site × position matrix ----------
    # By-trade lookup: (dept, pos) -> {site_id -> count}, and leave count.
    by_trade = defaultdict(lambda: defaultdict(int))
    leave_by_trade = defaultdict(int)
    dept_pos_seen = {}        # ordering: (dept, pos) -> first-insert index
    dept_seen = {}            # ordering: dept -> first-insert index

    for i, e in enumerate(emps_qs.values(
        'department', 'position', 'salary_grade', 'site_id', 'status'
    )):
        d, p = _resolve(e)
        key = (d, p)
        dept_seen.setdefault(d, i)
        dept_pos_seen.setdefault(key, i)
        if (e['status'] or '').strip() == 'Leave':
            leave_by_trade[key] += 1
            continue
        # site_id might be None for unassigned employees — those land in 'unassigned',
        # which we won't column-show but will still count in the row total.
        by_trade[key][e['site_id']] += 1

    depts = sorted(dept_seen.keys(), key=lambda x: x.lower())
    rows = []
    grand_total = 0
    for dept in depts:
        rows.append({'kind': 'dept', 'name': dept})
        positions = sorted(
            [k for k in dept_pos_seen.keys() if k[0] == dept],
            key=lambda x: x[1].lower(),
        )
        sub_counts = {sid: 0 for sid in site_ids}
        sub_leave = 0
        sub_total = 0
        for key in positions:
            (_, p) = key
            per_site = by_trade.get(key, {})
            counts = {sid: per_site.get(sid, 0) for sid in site_ids}
            leave = leave_by_trade.get(key, 0)
            total = sum(counts.values()) + leave
            grand_total += total
            sub_leave += leave
            sub_total += total
            for sid, n in counts.items():
                sub_counts[sid] += n
            rows.append({
                'kind': 'trade',
                'department': dept,
                'name': p,
                'counts': counts,
                'leave': leave,
                'total': total,
            })
        rows.append({
            'kind': 'subtotal',
            'department': dept,
            'counts': sub_counts,
            'leave': sub_leave,
            'total': sub_total,
        })

    return {
        'employee_type': employee_type,
        'sites': site_list,
        'rows': rows,
        'grand_total': grand_total,
        'selected_site': site_id_filter or 'all',
    }


class _AllAccessRequest:
    """A minimal request-like object so distribution snapshots are computed for
    ALL sites / ALL employees (superuser scope), independent of who triggers it."""
    class _U:
        is_superuser = True
        is_staff = True
        is_authenticated = True

    def __init__(self):
        self.user = self._U()
        self.GET = {}   # no 'site' filter → every site


def capture_distribution_snapshots(for_date=None):
    """Freeze today's (or for_date's) distribution for all three tab types.

    Stores the full all-sites payload so any past date can be reproduced exactly.
    Safe to call repeatedly — it upserts the row for that date+type.
    """
    for_date = for_date or timezone.localdate()
    shim = _AllAccessRequest()
    saved = 0
    for t in ('resource', 'staff', 'worker'):
        try:
            payload = _distribution_payload(shim, t)
            DistributionSnapshot.objects.update_or_create(
                date=for_date, dist_type=t, defaults={'payload': payload},
            )
            saved += 1
        except Exception:  # noqa: BLE001 — never let snapshotting break a page load
            logger.exception("capture_distribution_snapshots failed for type=%s", t)
    prune_distribution_snapshots()
    return saved


def prune_distribution_snapshots():
    """Delete snapshots older than the configured retention window.

    Controlled by AppSettings.distribution_snapshot_retention_months
    (0 = keep forever). Months are approximated as 30 days. Returns count deleted.
    """
    try:
        months = AppSettings.load().distribution_snapshot_retention_months
    except Exception:  # noqa: BLE001
        months = 12
    if not months or months <= 0:
        return 0   # unlimited retention
    from datetime import timedelta
    cutoff = timezone.localdate() - timedelta(days=int(months) * 30)
    deleted, _ = DistributionSnapshot.objects.filter(date__lt=cutoff).delete()
    return deleted


class ManpowerDistributionView(APIView):
    """GET /api/attendance/distribution/?type=staff|worker|resource (&site=ID&date=YYYY-MM-DD)

    No date (or today) → live data from the current roster, and today's snapshot
    is refreshed in passing so history accrues. A past date → the stored snapshot
    for that day (or the most recent snapshot on/before it), so admins can see what
    the distribution looked like back then.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        t = (request.GET.get('type') or 'staff').lower()
        if t not in ('staff', 'worker', 'resource'):
            return Response({'error': "type must be 'staff', 'worker' or 'resource'"}, status=400)

        today = timezone.localdate()
        date_str = (request.GET.get('date') or '').strip()
        req_date = None
        if date_str:
            try:
                from datetime import datetime as _dt
                req_date = _dt.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                req_date = None

        # Past-date view (historical) — superuser only, since snapshots hold all
        # sites and site-admin scoping can't be reapplied to a frozen payload.
        if req_date and req_date < today and request.user.is_superuser:
            snap = (DistributionSnapshot.objects
                    .filter(dist_type=t, date__lte=req_date)
                    .order_by('-date')
                    .first())
            if not snap:
                return Response({
                    'employee_type': t, 'sites': [], 'rows': [], 'grand_total': 0,
                    'historical': True, 'no_data': True,
                    'requested_date': str(req_date),
                    'message': 'No saved snapshot on or before this date. History is kept from the day this feature was enabled.',
                })
            data = dict(snap.payload or {})
            data['historical'] = True
            data['requested_date'] = str(req_date)
            data['snapshot_date'] = str(snap.date)
            return Response(data)

        # Live (current) view — unchanged behaviour, plus a passing snapshot
        # capture so today's history is recorded even without the cron job.
        data = _distribution_payload(request, t)
        try:
            DistributionSnapshot.objects.update_or_create(
                date=today, dist_type=t,
                defaults={'payload': _distribution_payload(_AllAccessRequest(), t)},
            )
        except Exception:  # noqa: BLE001
            logger.exception("distribution snapshot upsert failed")
        data['historical'] = False
        data['snapshot_date'] = str(today)
        return Response(data)


class ManpowerDistributionExportView(APIView):
    """GET /api/attendance/distribution/export/?type=...&site=...
    Excel download in the same row layout (Sr.No | Trade | <Site columns> | Leave | Total).
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    _TITLE = {
        'staff':    'PIC Group — DEPT-WISE / TRADE-WISE / PROJECT-WISE STRENGTH OF STAFF',
        'worker':   'PIC Group — DEPT-WISE / TRADE-WISE / PROJECT-WISE STRENGTH OF WORKERS',
        'resource': 'MANPOWER RESOURCES',
    }

    def get(self, request):
        import io
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from django.http import HttpResponse

        t = (request.GET.get('type') or 'staff').lower()
        if t not in ('staff', 'worker', 'resource'):
            return HttpResponse('Bad type', status=400)

        # Date-aware: a past date (superuser) exports the saved snapshot for that day.
        today = timezone.localdate()
        date_str = (request.GET.get('date') or '').strip()
        req_date = None
        if date_str:
            try:
                from datetime import datetime as _dt
                req_date = _dt.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                req_date = None
        as_of = today
        if req_date and req_date < today and request.user.is_superuser:
            snap = (DistributionSnapshot.objects
                    .filter(dist_type=t, date__lte=req_date)
                    .order_by('-date').first())
            if not snap:
                return HttpResponse('No saved snapshot on or before that date.', status=404)
            data = dict(snap.payload or {})
            as_of = snap.date
        else:
            data = _distribution_payload(request, t)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = {'staff': 'Staff', 'worker': 'Workers', 'resource': 'Resources'}[t]

        thin = Side(border_style='thin', color='999999')
        border = Border(top=thin, bottom=thin, left=thin, right=thin)
        title_font = Font(bold=True, size=12)
        header_fill = PatternFill(start_color='2563EB', end_color='2563EB', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')
        dept_fill = PatternFill(start_color='F3F4F6', end_color='F3F4F6', fill_type='solid')
        dept_font = Font(bold=True)
        subtotal_fill = PatternFill(start_color='FEF3C7', end_color='FEF3C7', fill_type='solid')
        subtotal_font = Font(bold=True)
        center = Alignment(horizontal='center', vertical='center', wrap_text=True)

        # Title row
        ws.cell(row=1, column=1, value=self._TITLE[t]).font = title_font
        # Date
        ws.cell(row=1, column=4, value=f"Date: {as_of.isoformat()}")

        # Resource sheet uses a flat layout (no site columns / no leave column)
        if t == 'resource':
            headers = ['Sr. No.', 'Department', 'Trade', 'Count']
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=3, column=c, value=h)
                cell.fill = header_fill; cell.font = header_font
                cell.alignment = center; cell.border = border
            r = 4
            sr = 1
            current_dept = ''
            for row in data['rows']:
                if row['kind'] == 'dept':
                    current_dept = row['name']
                    cell = ws.cell(row=r, column=1, value=current_dept)
                    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(headers))
                    cell.fill = dept_fill; cell.font = dept_font
                    r += 1
                elif row['kind'] == 'trade':
                    ws.cell(row=r, column=1, value=sr).border = border
                    ws.cell(row=r, column=2, value=row.get('department', '')).border = border
                    ws.cell(row=r, column=3, value=row['name']).border = border
                    ws.cell(row=r, column=4, value=row['count']).border = border
                    sr += 1; r += 1
                elif row['kind'] == 'subtotal':
                    ws.cell(row=r, column=3, value='TOTAL').font = subtotal_font
                    cell = ws.cell(row=r, column=4, value=row['count'])
                    cell.font = subtotal_font; cell.fill = subtotal_fill; cell.border = border
                    ws.cell(row=r, column=3).fill = subtotal_fill
                    ws.cell(row=r, column=3).border = border
                    r += 1
            ws.column_dimensions['B'].width = 32
            ws.column_dimensions['C'].width = 40
            ws.freeze_panes = 'A4'
            # Skip the rest of the staff/worker layout
            out = io.BytesIO()
            wb.save(out); out.seek(0)
            resp = HttpResponse(
                out.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
            resp['Content-Disposition'] = f'attachment; filename=manpower_resources_{timezone.localdate()}.xlsx'
            return resp

        # Staff / worker — dynamic site columns
        sites = data['sites']
        headers = ['Sr. No.', 'Trade'] + [s['name'] for s in sites] + ['Leave', 'Total']
        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=3, column=c, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
            cell.border = border

        r = 4
        sr = 1
        for row in data['rows']:
            if row['kind'] == 'dept':
                cell = ws.cell(row=r, column=1, value=row['name'])
                ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(headers))
                cell.fill = dept_fill
                cell.font = dept_font
                r += 1
            elif row['kind'] == 'trade':
                ws.cell(row=r, column=1, value=sr).border = border
                ws.cell(row=r, column=2, value=row['name']).border = border
                for ci, s in enumerate(sites, start=3):
                    ws.cell(row=r, column=ci, value=row['counts'].get(s['id'], 0)).border = border
                ws.cell(row=r, column=len(headers) - 1, value=row['leave']).border = border
                ws.cell(row=r, column=len(headers), value=row['total']).border = border
                sr += 1
                r += 1
            elif row['kind'] == 'subtotal':
                ws.cell(row=r, column=2, value='TOTAL').font = subtotal_font
                for ci, s in enumerate(sites, start=3):
                    cell = ws.cell(row=r, column=ci, value=row['counts'].get(s['id'], 0))
                    cell.font = subtotal_font
                    cell.fill = subtotal_fill
                    cell.border = border
                cell = ws.cell(row=r, column=len(headers) - 1, value=row['leave'])
                cell.font = subtotal_font; cell.fill = subtotal_fill; cell.border = border
                cell = ws.cell(row=r, column=len(headers), value=row['total'])
                cell.font = subtotal_font; cell.fill = subtotal_fill; cell.border = border
                ws.cell(row=r, column=2).fill = subtotal_fill
                ws.cell(row=r, column=2).border = border
                r += 1

        # Auto column widths
        for col_idx, h in enumerate(headers, start=1):
            letter = openpyxl.utils.get_column_letter(col_idx)
            ws.column_dimensions[letter].width = max(10, len(h) + 2)
        ws.column_dimensions['B'].width = 36
        ws.freeze_panes = 'C4'

        out = io.BytesIO()
        wb.save(out)
        out.seek(0)
        resp = HttpResponse(
            out.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        resp['Content-Disposition'] = f'attachment; filename=manpower_{t}_{timezone.localdate()}.xlsx'
        return resp


@login_required(login_url='admin-login')
def admin_distribution_list_view(request):
    """Renders the Distribution List page skeleton — the tabs populate via AJAX."""
    if not request.user.is_staff:
        return redirect('admin-login')
    is_superuser = request.user.is_superuser
    return render(request, 'distribution_list.html', {
        'is_superuser': is_superuser,
    })


# ---------------------------------------------------------------------------
# Department + Position management (CRUD)
# ---------------------------------------------------------------------------

def _norm_dept_name(s):
    """Normalize a department/position name — strip, collapse whitespace."""
    import re as _re
    if not s:
        return ''
    return _re.sub(r'\s+', ' ', str(s)).strip()


class AdminDepartmentsView(APIView):
    """List + create departments.

    GET  /api/attendance/admin-departments/
        → { departments: [{id, name, manager_name, is_active, positions_count}] }
    POST /api/attendance/admin-departments/
        body: {name, manager_name?}
        → 201 {id, name, manager_name, is_active}
    """
    permission_classes = [IsAdminUser]

    def get(self, request):
        rows = Department.objects.filter(is_active=True).order_by('sheet_order', 'name')
        items = []
        for d in rows:
            items.append({
                'id': d.id,
                'name': d.name,
                'manager_name': d.manager_name or '',
                'is_active': d.is_active,
                'positions_count': d.positions.filter(is_active=True).count(),
            })
        return Response({'count': len(items), 'departments': items})

    def post(self, request):
        name = _norm_dept_name(request.data.get('name'))
        if not name:
            return Response({'error': 'Department name is required.'}, status=400)
        # Case-insensitive dedupe. Revive a soft-deleted department instead of
        # erroring (its name is still in the table with is_active=False).
        existing = Department.objects.filter(name__iexact=name).first()
        if existing:
            if existing.is_active:
                return Response({'error': f'Department "{existing.name}" already exists.'}, status=400)
            existing.is_active = True
            if request.data.get('manager_name') is not None:
                existing.manager_name = _norm_dept_name(request.data.get('manager_name')) or None
            existing.save(update_fields=['is_active', 'manager_name'])
            return Response({
                'success': True, 'id': existing.id, 'name': existing.name,
                'manager_name': existing.manager_name or '', 'is_active': True,
            }, status=200)
        manager = _norm_dept_name(request.data.get('manager_name')) or None
        from django.db.models import Max as _Max
        order = Department.objects.aggregate(m=_Max('sheet_order'))['m'] or 0
        d = Department.objects.create(
            name=name, manager_name=manager,
            sheet_order=order + 1, is_active=True,
        )
        return Response({
            'success': True,
            'id': d.id, 'name': d.name,
            'manager_name': d.manager_name or '',
            'is_active': d.is_active,
        }, status=201)


class AdminDepartmentDetailView(APIView):
    """Update / delete one department.

    PUT    /api/attendance/admin-departments/<id>/    body: {name?, manager_name?}
    DELETE /api/attendance/admin-departments/<id>/    soft-deletes (is_active=False)
    """
    permission_classes = [IsAdminUser]

    def put(self, request, dept_id):
        try:
            d = Department.objects.get(id=dept_id)
        except Department.DoesNotExist:
            return Response({'error': 'Department not found'}, status=404)
        data = request.data
        if 'name' in data:
            new_name = _norm_dept_name(data.get('name'))
            if not new_name:
                return Response({'error': 'Name cannot be blank.'}, status=400)
            # Case-insensitive dedupe (excluding self).
            clash = Department.objects.filter(name__iexact=new_name).exclude(id=d.id).first()
            if clash:
                return Response({'error': f'Another department already uses "{clash.name}".'}, status=400)
            old_name = d.name
            d.name = new_name
            # Keep the denormalized JobCategory.department string in sync.
            JobCategory.objects.filter(department_fk=d).update(department=new_name)
            # Also keep any Employee.department strings in sync (case-insensitive).
            Employee.objects.filter(department__iexact=old_name).update(department=new_name)
        if 'manager_name' in data:
            d.manager_name = _norm_dept_name(data.get('manager_name')) or None
        d.save()
        return Response({
            'success': True,
            'id': d.id, 'name': d.name,
            'manager_name': d.manager_name or '',
            'is_active': d.is_active,
        })

    def delete(self, request, dept_id):
        try:
            d = Department.objects.get(id=dept_id)
        except Department.DoesNotExist:
            return Response({'error': 'Department not found'}, status=404)
        # Soft-delete: hide the department + its positions from dropdowns,
        # but leave Employee.department text intact (so historic data isn't lost).
        d.is_active = False
        d.save(update_fields=['is_active'])
        JobCategory.objects.filter(department_fk=d).update(is_active=False)
        return Response({'success': True})


class AdminPositionsView(APIView):
    """List + create positions.

    GET  /api/attendance/admin-positions/?department=<id>&employee_type=<staff|worker|resource>
        → { positions: [{id, name, employee_type, department_id, department_name}] }
    POST /api/attendance/admin-positions/
        body: {name, department_id, employee_type?}
        → 201 {id, name, ...}
    """
    permission_classes = [IsAdminUser]

    def get(self, request):
        qs = JobCategory.objects.filter(is_active=True).select_related('department_fk')
        dept_id = request.GET.get('department')
        if dept_id:
            try:
                qs = qs.filter(department_fk_id=int(dept_id))
            except (ValueError, TypeError):
                pass
        emp_type = (request.GET.get('employee_type') or '').strip().lower()
        if emp_type in ('staff', 'worker', 'resource'):
            qs = qs.filter(employee_type=emp_type)
        items = [{
            'id': p.id,
            'name': p.name,
            'employee_type': p.employee_type,
            'department_id': p.department_fk_id,
            'department_name': p.department_fk.name if p.department_fk else (p.department or ''),
        } for p in qs.order_by('sheet_order', 'name')]
        return Response({'count': len(items), 'positions': items})

    def post(self, request):
        name = _norm_dept_name(request.data.get('name'))
        if not name:
            return Response({'error': 'Position name is required.'}, status=400)
        dept_id = request.data.get('department_id')
        try:
            dept = Department.objects.get(id=dept_id) if dept_id else None
        except Department.DoesNotExist:
            return Response({'error': 'Department not found'}, status=400)
        emp_type = (request.data.get('employee_type') or 'worker').strip().lower()
        if emp_type not in ('staff', 'worker', 'resource'):
            emp_type = 'worker'
        # Case-insensitive dedupe within the same employee_type. The DB also has a
        # unique_together(name, employee_type), so even a SOFT-DELETED clash would
        # block a fresh insert. If the clash is active → error; if it was deleted →
        # revive it and reassign it to this department (that's what "add again" means).
        clash = JobCategory.objects.filter(name__iexact=name, employee_type=emp_type).first()
        if clash:
            if clash.is_active:
                where = f" under {clash.department_fk.name}" if clash.department_fk else ""
                return Response(
                    {'error': f'Position "{clash.name}" ({clash.get_employee_type_display()}) already exists{where}.'},
                    status=400,
                )
            clash.is_active = True
            clash.name = name
            clash.department = dept.name if dept else None
            clash.department_fk = dept
            clash.save(update_fields=['is_active', 'name', 'department', 'department_fk'])
            return Response({
                'success': True, 'id': clash.id, 'name': clash.name,
                'employee_type': clash.employee_type,
                'department_id': dept.id if dept else None,
                'department_name': dept.name if dept else '',
            }, status=200)
        from django.db.models import Max as _Max
        order = JobCategory.objects.aggregate(m=_Max('sheet_order'))['m'] or 0
        p = JobCategory.objects.create(
            name=name,
            department=dept.name if dept else None,
            department_fk=dept,
            employee_type=emp_type,
            sheet_order=order + 1,
            is_active=True,
        )
        return Response({
            'success': True,
            'id': p.id, 'name': p.name,
            'employee_type': p.employee_type,
            'department_id': dept.id if dept else None,
            'department_name': dept.name if dept else '',
        }, status=201)


class AdminPositionDetailView(APIView):
    """Update / delete one position.

    PUT    /api/attendance/admin-positions/<id>/  body: {name?, department_id?, employee_type?}
    DELETE /api/attendance/admin-positions/<id>/  soft-delete
    """
    permission_classes = [IsAdminUser]

    def put(self, request, pos_id):
        try:
            p = JobCategory.objects.get(id=pos_id)
        except JobCategory.DoesNotExist:
            return Response({'error': 'Position not found'}, status=404)
        data = request.data
        old_name = p.name
        if 'name' in data:
            new_name = _norm_dept_name(data.get('name'))
            if not new_name:
                return Response({'error': 'Name cannot be blank.'}, status=400)
            clash = JobCategory.objects.filter(name__iexact=new_name, employee_type=p.employee_type).exclude(id=p.id).first()
            if clash:
                return Response({'error': f'Another position already uses "{clash.name}".'}, status=400)
            p.name = new_name
            # Keep Employee.position in sync (case-insensitive).
            Employee.objects.filter(position__iexact=old_name).update(position=new_name)
        if 'department_id' in data:
            dept_id = data.get('department_id')
            if dept_id:
                try:
                    dept = Department.objects.get(id=dept_id)
                except Department.DoesNotExist:
                    return Response({'error': 'Department not found'}, status=400)
                p.department_fk = dept
                p.department = dept.name
            else:
                p.department_fk = None
                p.department = None
        if 'employee_type' in data:
            t = (data.get('employee_type') or '').strip().lower()
            if t in ('staff', 'worker', 'resource'):
                p.employee_type = t
        p.save()
        return Response({
            'success': True,
            'id': p.id, 'name': p.name,
            'employee_type': p.employee_type,
            'department_id': p.department_fk_id,
            'department_name': p.department_fk.name if p.department_fk else (p.department or ''),
        })

    def delete(self, request, pos_id):
        try:
            p = JobCategory.objects.get(id=pos_id)
        except JobCategory.DoesNotExist:
            return Response({'error': 'Position not found'}, status=404)
        p.is_active = False
        p.save(update_fields=['is_active'])
        return Response({'success': True})


@login_required(login_url='admin-login')
def admin_departments_management_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'departments_management.html', {
        'is_superuser': request.user.is_superuser,
    })


# ---------------------------------------------------------------------------
# AI-style analytics: Attrition risk + Document expiry + Excel auto-mapper
# ---------------------------------------------------------------------------

def _compute_attrition_score(emp, today,
                              attendance_recent_count,
                              attendance_baseline_count,
                              sick_count_60d,
                              geofence_misses_30d,
                              last_salary_days_ago):
    """Heuristic 0-100 risk score per employee + ordered list of contributing
    factors. Explainable on purpose: HR can read why a person is high-risk and
    act on it. Weights chosen so the worst plausible employee tops out near 100
    and the average healthy employee sits well under 20.
    """
    score = 0
    factors = []

    # --- 1. Attendance decline (max 30) -----------------------------------
    # Compare last 14d check-ins to the prior 30d baseline (days 14-44).
    # Drop > 30% = a real signal.
    recent_rate = attendance_recent_count / 14.0
    baseline_rate = attendance_baseline_count / 30.0 if attendance_baseline_count else 0
    if baseline_rate > 0 and recent_rate < baseline_rate * 0.70:
        drop_pct = (baseline_rate - recent_rate) / baseline_rate
        pts = min(30, int(drop_pct * 40))
        if pts > 0:
            score += pts
            factors.append({
                'label': f'Attendance down {int(drop_pct*100)}% vs prior month',
                'points': pts,
                'severity': 'high' if pts >= 15 else 'med',
            })

    # --- 2. Sick-leave frequency (max 20) ---------------------------------
    if sick_count_60d >= 3:
        pts = min(20, sick_count_60d * 4)
        score += pts
        factors.append({
            'label': f'{sick_count_60d} sick days in last 60 days',
            'points': pts,
            'severity': 'high' if sick_count_60d >= 5 else 'med',
        })

    # --- 3. Geofence misses (max 15) --------------------------------------
    if geofence_misses_30d >= 3:
        pts = min(15, geofence_misses_30d * 2)
        score += pts
        factors.append({
            'label': f'{geofence_misses_30d} geofence misses (last 30d)',
            'points': pts,
            'severity': 'med',
        })

    # --- 4. Stale salary (max 20) -----------------------------------------
    # Past 12 months without a salary change is mildly concerning;
    # past 18+ months is a strong demotivator signal.
    if last_salary_days_ago is not None and last_salary_days_ago > 365:
        months_stale = (last_salary_days_ago - 365) // 30
        pts = min(20, months_stale * 2)
        if pts > 0:
            score += pts
            factors.append({
                'label': f'No salary change in {last_salary_days_ago // 30} months',
                'points': pts,
                'severity': 'high' if months_stale >= 6 else 'med',
            })

    # --- 5. Tenure inverse (max 15) ---------------------------------------
    # Newer employees (< 12 mo tenure) are statistically higher flight risk.
    if emp.date_of_joining:
        tenure_days = (today - emp.date_of_joining).days
        if 0 <= tenure_days < 365:
            tenure_months = tenure_days // 30
            pts = max(0, 15 - tenure_months)
            if pts > 0:
                score += pts
                factors.append({
                    'label': f'Tenure only {tenure_months} months',
                    'points': pts,
                    'severity': 'med' if pts >= 10 else 'low',
                })

    return min(100, score), factors


class AttritionRiskView(APIView):
    """GET /api/attendance/analytics/attrition-risk/?site=&department=&min_score=
    Returns Active employees ranked by attrition-risk score with explainable factors.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from collections import Counter as _Counter
        from datetime import timedelta

        today = timezone.localdate()
        scope = Employee.objects.select_related('site').filter(status__iexact='Active')

        # Optional filters (mirror the dashboard semantics)
        site_id = request.GET.get('site')
        if site_id and site_id != 'all':
            try:
                scope = scope.filter(site_id=int(site_id))
            except (ValueError, TypeError):
                pass
        dept = (request.GET.get('department') or '').strip()
        if dept and dept != 'all':
            scope = scope.filter(department__iexact=dept)
        try:
            min_score = int(request.GET.get('min_score') or 0)
        except (TypeError, ValueError):
            min_score = 0

        # Site-admin scope
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                scope = scope.filter(site__in=profile.sites.all())
            except AdminProfile.DoesNotExist:
                scope = scope.none()

        emp_ids = list(scope.values_list('id', flat=True))
        if not emp_ids:
            return Response({'count': 0, 'rows': []})

        # ── Batch the per-employee aggregates so we don't make N queries ─
        recent_window_start = today - timedelta(days=14)
        baseline_start = today - timedelta(days=44)
        sick_window_start = today - timedelta(days=60)
        geo_window_start = today - timedelta(days=30)

        # Attendance counts per employee (recent 14d)
        recent_qs = (Attendance.objects
                     .filter(user_id__in=emp_ids, date__gte=recent_window_start, date__lte=today)
                     .values('user_id').annotate(n=Count('id')))
        recent_map = {r['user_id']: r['n'] for r in recent_qs}

        # Attendance counts per employee (prior 30d baseline = days 14..44)
        baseline_qs = (Attendance.objects
                       .filter(user_id__in=emp_ids,
                               date__gte=baseline_start, date__lt=recent_window_start)
                       .values('user_id').annotate(n=Count('id')))
        baseline_map = {r['user_id']: r['n'] for r in baseline_qs}

        # Sick count per employee (60d)
        sick_qs = (Attendance.objects
                   .filter(user_id__in=emp_ids, status='sick', date__gte=sick_window_start)
                   .values('user_id').annotate(n=Count('id')))
        sick_map = {r['user_id']: r['n'] for r in sick_qs}

        # Geofence misses per employee (30d)
        geo_qs = (Attendance.objects
                  .filter(user_id__in=emp_ids, is_within_geofence=False, date__gte=geo_window_start)
                  .values('user_id').annotate(n=Count('id')))
        geo_map = {r['user_id']: r['n'] for r in geo_qs}

        # Latest salary change date per employee
        salary_qs = (EmployeeSalaryHistory.objects
                     .filter(employee_id__in=emp_ids)
                     .values('employee_id'))
        latest_salary_map = {}
        for r in EmployeeSalaryHistory.objects.filter(employee_id__in=emp_ids).order_by('employee_id', '-effective_from'):
            if r.employee_id not in latest_salary_map:
                latest_salary_map[r.employee_id] = r.effective_from

        # ── Score every employee ─────────────────────────────────────────
        rows = []
        for emp in scope:
            last_salary_days = None
            if emp.id in latest_salary_map:
                last_salary_days = (today - latest_salary_map[emp.id]).days
            elif emp.date_of_joining:
                last_salary_days = (today - emp.date_of_joining).days

            score, factors = _compute_attrition_score(
                emp, today,
                attendance_recent_count=recent_map.get(emp.id, 0),
                attendance_baseline_count=baseline_map.get(emp.id, 0),
                sick_count_60d=sick_map.get(emp.id, 0),
                geofence_misses_30d=geo_map.get(emp.id, 0),
                last_salary_days_ago=last_salary_days,
            )
            if score < min_score:
                continue
            rows.append({
                'employee_id': emp.id,
                'name': emp.name,
                'badge_number': emp.badge_number or '',
                'site': emp.site.name if emp.site else '',
                'department': emp.department or '',
                'position': emp.position or '',
                'score': score,
                'band': 'high' if score >= 50 else ('medium' if score >= 25 else 'low'),
                'factors': factors,
            })

        rows.sort(key=lambda r: r['score'], reverse=True)

        # Distribution summary for the page header pills
        band_counts = _Counter(r['band'] for r in rows)

        return Response({
            'count': len(rows),
            'as_of': str(today),
            'bands': {'high': band_counts.get('high', 0),
                      'medium': band_counts.get('medium', 0),
                      'low': band_counts.get('low', 0)},
            'rows': rows,
        })


# ---------------------------------------------------------------------------
# Document expiry
# ---------------------------------------------------------------------------

def _doc_bucket(days):
    if days is None: return None
    if days < 0:     return 'expired'
    if days <= 14:   return 'critical'
    if days <= 30:   return 'urgent'
    if days <= 90:   return 'soon'
    return 'ok'


class DocumentExpiryView(APIView):
    """GET /api/attendance/analytics/document-expiry/?site=&bucket=
    Returns Active employees with passport or visa expiring soon (or already
    expired), bucketed by urgency.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        today = timezone.localdate()
        scope = Employee.objects.select_related('site').filter(status__iexact='Active')

        site_id = request.GET.get('site')
        if site_id and site_id != 'all':
            try:
                scope = scope.filter(site_id=int(site_id))
            except (ValueError, TypeError):
                pass

        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                scope = scope.filter(site__in=profile.sites.all())
            except AdminProfile.DoesNotExist:
                scope = scope.none()

        bucket_filter = (request.GET.get('bucket') or '').strip().lower() or None

        rows = []
        bucket_counts = {'expired': 0, 'critical': 0, 'urgent': 0, 'soon': 0}
        for emp in scope:
            docs = []
            worst = None
            if emp.passport_expiry:
                d = (emp.passport_expiry - today).days
                b = _doc_bucket(d)
                docs.append({'type': 'Passport', 'date': str(emp.passport_expiry), 'days': d, 'bucket': b})
            if emp.visa_expiry_date:
                d = (emp.visa_expiry_date - today).days
                b = _doc_bucket(d)
                docs.append({'type': 'Visa', 'date': str(emp.visa_expiry_date), 'days': d, 'bucket': b})
            # Pick the worst doc bucket for this employee
            severity_order = ['expired', 'critical', 'urgent', 'soon', 'ok']
            for sev in severity_order:
                if any(doc['bucket'] == sev for doc in docs):
                    worst = sev
                    break
            if not worst or worst == 'ok':
                continue
            if bucket_filter and worst != bucket_filter:
                continue
            bucket_counts[worst] = bucket_counts.get(worst, 0) + 1
            rows.append({
                'employee_id': emp.id,
                'name': emp.name,
                'badge_number': emp.badge_number or '',
                'site': emp.site.name if emp.site else '',
                'department': emp.department or '',
                'docs': docs,
                'worst_bucket': worst,
            })

        # Sort: expired first, then by min days remaining
        sev_rank = {'expired': 0, 'critical': 1, 'urgent': 2, 'soon': 3}
        def _row_key(r):
            mind = min((d['days'] for d in r['docs'] if d['days'] is not None), default=99999)
            return (sev_rank.get(r['worst_bucket'], 9), mind)
        rows.sort(key=_row_key)

        return Response({
            'count': len(rows),
            'as_of': str(today),
            'buckets': bucket_counts,
            'rows': rows,
        })


# ---------------------------------------------------------------------------
# Excel auto-mapper preview
# ---------------------------------------------------------------------------

# Canonical field → alias keywords (lowercased, token-style).
# When the admin's sheet has weird columns, we score each header against this
# dictionary using token overlap + substring; the best match wins. Beats
# the rigid regex matcher in import_employees.
IMPORT_FIELD_ALIASES = {
    'name':             ['name', 'full name', 'employee name', 'worker name', 'worker', 'employee'],
    'badge_number':     ['badge', 'badge id', 'badge number', 'emp id', 'employee id', 'staff id', 'card id'],
    'mol_id':           ['mol', 'mol id', 'personal nr', 'personal number'],
    'labor_card_number':['labour card', 'labor card', 'l.card', 'cec', 'cec nr', 'work permit'],
    'passport_number':  ['passport', 'passport no', 'passport number', 'pp no', 'pp number'],
    'passport_expiry':  ['passport expiry', 'expiry date', 'pp expiry'],
    'visa_details':     ['visa', 'visa details', 'visa info'],
    'visa_expiry_date': ['visa expiry', 'visa expiry date'],
    'department':       ['department', 'dept', 'division', 'section'],
    'position':         ['position', 'designation', 'job title', 'present designation', 'trade'],
    'salary_grade':     ['category', 'grade', 'salary grade', 'level'],
    'site':             ['site', 'project', 'location'],
    'status':           ['status', 'employment status', 'active'],
    'nationality':      ['nationality', 'country'],
    'gender':           ['gender', 'sex'],
    'marital_status':   ['marital', 'marital status'],
    'religion':         ['religion'],
    'dob':              ['dob', 'date of birth', 'birth date', 'birthday'],
    'doj':              ['doj', 'date of joining', 'joining date', 'date joined', 'd.o.j', 'joined'],
    'gross_salary':     ['gross salary', 'gross', 'total salary'],
    'basic_salary':     ['basic salary', 'basic', 'base salary'],
    'job_description':  ['job description', 'job desc', 'description'],
    'sponsor':          ['sponsor', 'employer', 'sponsoring company'],
    'employer':         ['employer company', 'parent company', 'company', 'group'],
    'phone':            ['phone', 'mobile', 'contact'],
    'email':            ['email', 'e-mail'],
}


def _norm_header(s):
    import re as _re
    if s is None:
        return ''
    return _re.sub(r'[^a-z0-9 ]+', ' ', _re.sub(r'\s+', ' ', str(s).lower())).strip()


def _score_header_against_field(header_norm, aliases):
    """Returns 0-100 score for how well header matches any of the aliases.

    Tie-breaks substring matches by coverage — a longer matched alias wins
    over a shorter one, so "passport expiry date" picks `passport_expiry`
    instead of tying with the bare `passport` alias on `passport_number`.
    """
    if not header_norm:
        return 0
    h_tokens = set(header_norm.split())
    h_len = max(len(header_norm), 1)
    best = 0
    for alias in aliases:
        a = _norm_header(alias)
        if not a:
            continue
        if header_norm == a:
            return 100
        a_tokens = set(a.split())
        # Substring boost — weighted by how much of the header the alias
        # actually covers, so longer aliases outrank shorter ones.
        if a in header_norm:
            coverage = len(a) / h_len
            best = max(best, 60 + int(coverage * 35))   # range 60-95
            continue
        if header_norm in a:
            coverage = h_len / max(len(a), 1)
            best = max(best, 60 + int(coverage * 25))   # range 60-85
            continue
        # Token-overlap (Jaccard-style) fallback
        if h_tokens and a_tokens:
            inter = len(h_tokens & a_tokens)
            if inter:
                jaccard = inter / max(len(h_tokens | a_tokens), 1)
                best = max(best, int(jaccard * 80))
    return best


class ExcelMappingPreviewView(APIView):
    """POST /api/attendance/employees/import-preview/

    Body: multipart form with `file` = the Excel sheet.
    Returns the auto-detected column mapping (with confidence) so the admin
    can confirm or override BEFORE the actual import runs. No DB writes here.
    """
    permission_classes = [IsAdminUser]

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'No file uploaded.'}, status=400)
        try:
            df = pd.read_excel(file, header=None, engine='openpyxl')
        except Exception as e:  # noqa: BLE001
            return Response({'error': f'Could not read Excel: {e}'}, status=400)

        # Find the most-likely header row in the first 20 rows.
        header_idx = -1
        for i in range(min(20, len(df))):
            row = df.iloc[i].astype(str).str.strip().tolist()
            if any(any(k in v.lower() for k in ('name', 'badge', 'status', 'sr.', 'sl.no')) for v in row if v != 'nan'):
                header_idx = i
                break
        if header_idx == -1:
            return Response({'error': "Couldn't find a header row in the first 20 rows."}, status=400)

        headers = [str(v).strip() if v is not None else '' for v in df.iloc[header_idx].tolist()]
        # Also peek at the row below (merged-header sheets often split labels across two rows)
        sub_headers = []
        if header_idx + 1 < len(df):
            sub_headers = [str(v).strip() if v is not None else '' for v in df.iloc[header_idx + 1].tolist()]

        mapping = []
        for col_idx, h in enumerate(headers):
            text = h
            if sub_headers and col_idx < len(sub_headers) and sub_headers[col_idx] and sub_headers[col_idx].lower() != 'nan':
                text = f'{h} {sub_headers[col_idx]}'.strip()
            text_norm = _norm_header(text)
            if not text_norm or text_norm == 'nan':
                continue
            best_field, best_score = None, 0
            for field, aliases in IMPORT_FIELD_ALIASES.items():
                score = _score_header_against_field(text_norm, aliases)
                if score > best_score:
                    best_field, best_score = field, score
            mapping.append({
                'column_index': col_idx,
                'header': text.strip(),
                'matched_field': best_field if best_score >= 40 else None,
                'confidence': best_score,
                'status': ('strong' if best_score >= 80
                           else 'likely' if best_score >= 60
                           else 'weak' if best_score >= 40
                           else 'unknown'),
            })

        return Response({
            'header_row': header_idx,
            'columns': mapping,
            'available_fields': sorted(IMPORT_FIELD_ALIASES.keys()),
        })


# ---------------------------------------------------------------------------
# Analytics template views
# ---------------------------------------------------------------------------

@login_required(login_url='admin-login')
def admin_attrition_risk_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'attrition_risk.html', {'is_superuser': request.user.is_superuser})


@login_required(login_url='admin-login')
def admin_document_expiry_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'document_expiry.html', {'is_superuser': request.user.is_superuser})


# ---------------------------------------------------------------------------
# Manpower distribution recommendations
# ---------------------------------------------------------------------------

class ManpowerRecommendationsView(APIView):
    """GET /api/attendance/analytics/manpower-recommendations/

    For each (department, position) combination, compares each site's headcount
    to the median across all sites that host the position. Flags sites that
    sit > median * `over_factor` or < median * `under_factor` and proposes
    redeployment moves from over → under.

    Query params:
        over_factor     — default 1.5  (over-staffed if count > median * 1.5)
        under_factor    — default 0.5  (under-staffed if count < median * 0.5)
        min_total       — default 4    (skip positions with < 4 total to avoid noise)
        site            — optional ID to filter recommendations involving that site
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from collections import defaultdict

        def _float(name, default):
            try:
                v = float(request.GET.get(name) or default)
                return max(0.01, v)
            except (TypeError, ValueError):
                return default
        def _int(name, default):
            try:
                return max(0, int(request.GET.get(name) or default))
            except (TypeError, ValueError):
                return default

        over_factor  = _float('over_factor', 1.5)
        under_factor = _float('under_factor', 0.5)
        min_total    = _int('min_total', 4)
        site_focus   = request.GET.get('site')

        # ── Pull every Active employee with site / department / position ──
        scope = (Employee.objects
                 .filter(status__iexact='Active')
                 .select_related('site')
                 .values('site_id', 'site__name', 'department', 'position', 'salary_grade'))

        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                allowed = set(profile.sites.values_list('id', flat=True))
                scope = [e for e in scope if e['site_id'] in allowed]
            except AdminProfile.DoesNotExist:
                scope = []

        # ── Group: (department, position) → site_id → count ──
        # Position uses Employee.position first, salary_grade fall-back
        # (matches how the distribution list counts).
        groups = defaultdict(lambda: defaultdict(int))  # (dept, pos) -> {site_id: n}
        site_names = {}
        for e in scope:
            if e.get('site_id') is None:
                continue
            dept = (e.get('department') or '—').strip() or '—'
            pos  = ((e.get('position') or e.get('salary_grade') or '').strip() or '—')
            if pos == '—':
                continue
            groups[(dept, pos)][e['site_id']] += 1
            site_names[e['site_id']] = e.get('site__name') or f'Site {e["site_id"]}'

        # ── Build recommendations per (dept, pos) ──
        recommendations = []
        for (dept, pos), per_site in groups.items():
            counts = list(per_site.values())
            total  = sum(counts)
            if total < min_total or len(counts) < 2:
                continue
            counts_sorted = sorted(counts)
            mid = len(counts_sorted) // 2
            median = (counts_sorted[mid] if len(counts_sorted) % 2
                      else (counts_sorted[mid - 1] + counts_sorted[mid]) / 2)
            if median <= 0:
                continue

            over  = []   # (site_id, count, surplus)
            under = []   # (site_id, count, deficit)
            for sid, n in per_site.items():
                if n > median * over_factor:
                    over.append((sid, n, n - median))
                elif n < median * under_factor:
                    under.append((sid, n, median - n))

            if not over or not under:
                continue

            over.sort(key=lambda x: -x[2])
            under.sort(key=lambda x: -x[2])

            # Greedy assignment from biggest surplus to biggest deficit
            moves = []
            i = j = 0
            while i < len(over) and j < len(under):
                src_id, src_n, src_surplus = over[i]
                dst_id, dst_n, dst_deficit = under[j]
                qty = max(1, int(min(src_surplus, dst_deficit)))
                if qty == 0:
                    break
                moves.append({
                    'from_site_id': src_id,
                    'from_site':    site_names[src_id],
                    'from_count':   src_n,
                    'to_site_id':   dst_id,
                    'to_site':      site_names[dst_id],
                    'to_count':     dst_n,
                    'quantity':     qty,
                })
                src_surplus -= qty
                dst_deficit -= qty
                over[i]  = (src_id, src_n, src_surplus)
                under[j] = (dst_id, dst_n, dst_deficit)
                if src_surplus <= 0: i += 1
                if dst_deficit <= 0: j += 1

            if site_focus and site_focus != 'all':
                try:
                    sf = int(site_focus)
                    moves = [m for m in moves if m['from_site_id'] == sf or m['to_site_id'] == sf]
                except (TypeError, ValueError):
                    pass
                if not moves:
                    continue

            recommendations.append({
                'department':   dept,
                'position':     pos,
                'total':        total,
                'median':       round(float(median), 1),
                'moves':        moves,
                'impact':       sum(m['quantity'] for m in moves),
            })

        recommendations.sort(key=lambda r: -r['impact'])

        return Response({
            'count':            len(recommendations),
            'as_of':            str(timezone.localdate()),
            'over_factor':      over_factor,
            'under_factor':     under_factor,
            'recommendations':  recommendations,
        })


# ---------------------------------------------------------------------------
# Executive summary (5 NL bullets)
# ---------------------------------------------------------------------------

class ExecutiveSummaryView(APIView):
    """GET /api/attendance/analytics/executive-summary/?period=daily|weekly

    Auto-composes 5 bullets summarising the state of the operation. Pure
    template-based NLG — no LLM call needed. Each bullet has a metric value,
    a comparison delta, and a sentence ready for display.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from datetime import timedelta
        from collections import defaultdict

        period = (request.GET.get('period') or 'daily').lower()
        if period not in ('daily', 'weekly'):
            period = 'daily'
        today = timezone.localdate()

        # Scope
        emp_scope = Employee.objects.all()
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                emp_scope = emp_scope.filter(site__in=profile.sites.all())
            except AdminProfile.DoesNotExist:
                emp_scope = emp_scope.none()

        active_emp_qs = emp_scope.filter(status__iexact='Active')
        active_emp_ids = list(active_emp_qs.values_list('id', flat=True))
        bullets = []

        # ── 1. Alerts today vs yesterday ─────────────────────────────────
        def _alerts_for(d):
            geofence = Attendance.objects.filter(date=d, is_within_geofence=False, user_id__in=active_emp_ids).count()
            on_leave = (Attendance.objects.filter(date=d, user__status='Leave', user_id__in=active_emp_ids)
                        .filter(Q(check_in_time__isnull=False) | Q(check_out_time__isnull=False))
                        .filter(Q(user__leave_start_date__isnull=True) | Q(user__leave_start_date__lte=d))
                        .filter(Q(user__leave_end_date__isnull=True)   | Q(user__leave_end_date__gte=d))
                        .count())
            return geofence + on_leave

        if period == 'daily':
            today_alerts = _alerts_for(today)
            comp_alerts  = _alerts_for(today - timedelta(days=1))
            comp_label   = 'yesterday'
        else:
            today_alerts = sum(_alerts_for(today - timedelta(days=i)) for i in range(7))
            comp_alerts  = sum(_alerts_for(today - timedelta(days=i)) for i in range(7, 14))
            comp_label   = 'previous 7 days'
        delta = today_alerts - comp_alerts
        pct = int(round((delta / comp_alerts) * 100)) if comp_alerts else None
        if delta > 0:
            sev = 'warn' if (pct is None or pct < 25) else 'bad'
            sentence = f"{today_alerts} alerts {'today' if period=='daily' else 'this week'} — "
            sentence += f"+{delta} vs {comp_label}" + (f" ({pct}% up)" if pct is not None else '')
        elif delta < 0:
            sev = 'good'
            sentence = f"{today_alerts} alerts {'today' if period=='daily' else 'this week'} — {abs(delta)} fewer than {comp_label}"
        else:
            sev = 'neutral'
            sentence = f"{today_alerts} alerts {'today' if period=='daily' else 'this week'} — unchanged vs {comp_label}"
        bullets.append({'icon': '🚨', 'severity': sev, 'metric': today_alerts, 'delta': delta, 'text': sentence})

        # ── 2. Biggest absence spike per site ─────────────────────────────
        # For each site, compare today's absent count (active employees w/ no
        # attendance) to the recent average.
        def _absent_for(d):
            present_ids = set(Attendance.objects.filter(date=d, user_id__in=active_emp_ids)
                              .values_list('user_id', flat=True))
            absent_by_site = defaultdict(int)
            for emp in active_emp_qs.values('id', 'site_id', 'site__name'):
                if emp['site_id'] is None or emp['id'] in present_ids:
                    continue
                absent_by_site[(emp['site_id'], emp.get('site__name') or 'Unassigned')] += 1
            return absent_by_site

        today_absent = _absent_for(today)
        # Avg over last 7 days (excluding today)
        baseline_totals = defaultdict(list)
        for back in range(1, 8):
            day_absent = _absent_for(today - timedelta(days=back))
            for site_key, n in day_absent.items():
                baseline_totals[site_key].append(n)
        spike = None  # (site_name, today_n, baseline_avg, delta)
        for site_key, n in today_absent.items():
            samples = baseline_totals.get(site_key, [])
            avg = sum(samples) / len(samples) if samples else 0
            d = n - avg
            if d >= 5 and (spike is None or d > spike[3]):
                spike = (site_key[1], n, avg, d)
        if spike:
            site_name, n, avg, d = spike
            bullets.append({'icon': '📍', 'severity': 'warn' if d < 15 else 'bad',
                            'metric': n,
                            'delta': round(d, 1),
                            'text': f"Absence spike at {site_name} — {n} absent today vs ~{int(round(avg))} typical (+{int(round(d))})"})
        else:
            bullets.append({'icon': '📍', 'severity': 'good', 'metric': 0, 'delta': 0,
                            'text': 'No site is significantly above its usual absence rate today.'})

        # ── 3. Document expiry urgency ────────────────────────────────────
        expired = critical = urgent = 0
        for emp in active_emp_qs.values('passport_expiry', 'visa_expiry_date'):
            for fld in ('passport_expiry', 'visa_expiry_date'):
                v = emp.get(fld)
                if not v:
                    continue
                d = (v - today).days
                if d < 0:    expired  += 1
                elif d <= 14: critical += 1
                elif d <= 30: urgent   += 1
        if expired or critical:
            bullets.append({'icon': '📄', 'severity': 'bad' if expired else 'warn',
                            'metric': expired + critical,
                            'delta': None,
                            'text': (f"{expired} document(s) already expired"
                                     + (f" and {critical} expiring in the next 14 days" if critical else ''))})
        elif urgent:
            bullets.append({'icon': '📄', 'severity': 'warn',
                            'metric': urgent, 'delta': None,
                            'text': f"{urgent} document(s) expiring within 30 days."})
        else:
            bullets.append({'icon': '📄', 'severity': 'good', 'metric': 0, 'delta': None,
                            'text': 'No documents expiring in the next 30 days.'})

        # ── 4. Manpower imbalance (most critically understaffed trade-site) ─
        # Reuses the same median-based detector as ManpowerRecommendationsView.
        groups = defaultdict(lambda: defaultdict(int))
        for e in active_emp_qs.values('site_id', 'site__name', 'department', 'position', 'salary_grade'):
            if e.get('site_id') is None:
                continue
            pos = ((e.get('position') or e.get('salary_grade') or '').strip())
            if not pos:
                continue
            groups[(e.get('department') or '—', pos)][(e['site_id'], e.get('site__name') or 'Unassigned')] += 1
        worst_under = None  # (dept, pos, site_name, count, median, deficit)
        for (dept, pos), per_site in groups.items():
            if len(per_site) < 2:
                continue
            counts = sorted(per_site.values())
            mid = len(counts) // 2
            median = counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2
            if median < 2:
                continue
            for (sid, sname), n in per_site.items():
                if n < median * 0.5:
                    deficit = median - n
                    if worst_under is None or deficit > worst_under[5]:
                        worst_under = (dept, pos, sname, n, median, deficit)
        if worst_under:
            dept, pos, sname, n, median, deficit = worst_under
            bullets.append({'icon': '👷', 'severity': 'warn' if deficit < 10 else 'bad',
                            'metric': int(deficit),
                            'delta': None,
                            'text': f"{sname} is short on {pos} — {int(n)} on site vs ~{int(round(median))} typical (need {int(round(deficit))} more)"})
        else:
            bullets.append({'icon': '👷', 'severity': 'good', 'metric': 0, 'delta': None,
                            'text': 'Manpower is well-balanced across sites today.'})

        # ── 5. Attendance rate snapshot ───────────────────────────────────
        today_attended = Attendance.objects.filter(date=today, user_id__in=active_emp_ids).values_list('user_id', flat=True).distinct().count()
        total_active = len(active_emp_ids)
        rate = (today_attended / total_active * 100) if total_active else 0
        last_week_rates = []
        for back in range(1, 8):
            d = today - timedelta(days=back)
            attended = Attendance.objects.filter(date=d, user_id__in=active_emp_ids).values_list('user_id', flat=True).distinct().count()
            last_week_rates.append((attended / total_active * 100) if total_active else 0)
        avg_rate = sum(last_week_rates) / len(last_week_rates) if last_week_rates else 0
        delta = rate - avg_rate
        if delta < -3:
            sev = 'warn' if delta > -8 else 'bad'
            txt = f"Attendance {rate:.1f}% — {abs(delta):.1f} pts below 7-day average ({avg_rate:.1f}%)"
        elif delta > 3:
            sev = 'good'
            txt = f"Attendance {rate:.1f}% — {delta:.1f} pts above 7-day average ({avg_rate:.1f}%)"
        else:
            sev = 'neutral'
            txt = f"Attendance {rate:.1f}% — in line with 7-day average ({avg_rate:.1f}%)"
        bullets.append({'icon': '📊', 'severity': sev, 'metric': round(rate, 1), 'delta': round(delta, 1), 'text': txt})

        # ── Optional LLM polish ──────────────────────────────────────────
        # Opt-in per request via ?ai=true. We compute the bullets with the
        # heuristic engine FIRST (so the response is correct even if the LLM
        # fails or is disabled), then rewrite the prose only if asked.
        engine = 'heuristic'
        if (request.GET.get('ai') or '').lower() in ('1', 'true', 'yes'):
            from attendance.services import llm as _llm
            polished = _llm.polish_summary(bullets)
            if polished:
                bullets = polished
                engine = 'llm'

        return Response({
            'as_of': str(today),
            'period': period,
            'engine': engine,
            'bullets': bullets,
        })


# ---------------------------------------------------------------------------
# Ask-the-Data — natural-language Q&A
# ---------------------------------------------------------------------------

class AskDataView(APIView):
    """POST /api/attendance/analytics/ask/   body: {"q": "<question>"}

    Two-tier NL parser:
      1) If AI_ENABLED + OPENAI_API_KEY are set, gpt-4o-mini parses the
         question into structured fields (much better at long-tail phrasings).
      2) On any LLM failure (disabled, missing key, timeout, malformed JSON,
         etc.) it falls back to the original keyword parser — predictable,
         free, never breaks.

    Returns the same `as_understood` shape either way + an `engine` field so
    the UI can show whether the answer came from the LLM or the heuristic.
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def post(self, request):
        from datetime import timedelta
        import re as _re
        from attendance.services import llm as _llm

        q_raw = (request.data.get('q') or '').strip()
        if not q_raw:
            return Response({'error': 'Please type a question.'}, status=400)
        today = timezone.localdate()

        # ── Snapshot the entity catalogues once (used by both parsers) ───
        sites_qs       = list(Site.objects.values('id', 'name'))
        positions_qs   = list(JobCategory.objects.filter(is_active=True).values_list('name', flat=True))
        departments_qs = list(Department.objects.filter(is_active=True).values_list('name', flat=True))

        # ── Engine 1: LLM parser (preferred when enabled) ────────────────
        engine = 'heuristic'
        parsed = None
        if _llm.ai_is_enabled():
            parsed = _llm.parse_question(
                q_raw,
                sites=[s['name'] for s in sites_qs],
                positions=positions_qs,
                departments=departments_qs,
            )
            if parsed:
                engine = 'llm'

        # ── Engine 2 (fallback): original keyword parser ─────────────────
        if not parsed:
            q = ' ' + q_raw.lower() + ' '
            intent = 'list'
            if any(k in q for k in (' how many ', ' how much ', ' count ', ' number of ', ' total ')):
                intent = 'count'
            if any(k in q for k in (' show me ', ' list ', ' find ', ' give me ')):
                intent = 'list'

            status = None
            if   ' on leave '   in q or ' on-leave '   in q or ' leave '       in q: status = 'Leave'
            elif ' resigned '   in q: status = 'Resigned'
            elif ' terminated ' in q: status = 'Terminated'
            elif ' active '     in q: status = 'Active'
            elif ' no renewal ' in q or ' no-renewal ' in q:                        status = 'No Renewal'
            elif ' absconding ' in q or ' absconded '  in q:                        status = 'Absconding'

            time_window = None
            if   ' today '     in q: time_window = 'today'
            elif ' yesterday ' in q: time_window = 'yesterday'
            elif ' this week ' in q: time_window = 'this_week'
            elif ' last week ' in q: time_window = 'last_week'
            elif ' this month ' in q: time_window = 'this_month'

            attendance_mod = None
            if ' absent ' in q:    attendance_mod = 'absent'
            elif ' present ' in q: attendance_mod = 'present'
            elif ' late ' in q:    attendance_mod = 'late'

            def _norm(s): return _re.sub(r'\s+', ' ', s.lower()).strip()
            q_norm = _norm(q_raw)
            site_name = next((s['name'] for s in sites_qs if _norm(s['name']) in q_norm), None)
            position  = next((p for p in positions_qs if len(p) >= 4 and _norm(p) in q_norm), None)
            department = next((d for d in departments_qs if _norm(d) in q_norm), None)

            parsed = {
                'intent': intent, 'status': status, 'attendance': attendance_mod,
                'site': site_name, 'position': position, 'department': department,
                'time_window': time_window,
            }

        # ── Resolve site name → id (works for both engines) ──────────────
        site = None
        if parsed.get('site'):
            site = next((s for s in sites_qs
                         if (s['name'] or '').strip().lower() == (parsed['site'] or '').strip().lower()),
                        None)

        intent         = parsed.get('intent') or 'list'
        status         = parsed.get('status')
        attendance_mod = parsed.get('attendance')
        position       = parsed.get('position')
        department     = parsed.get('department')

        # Translate the LLM's symbolic time-window into a date range
        date_filter = None
        tw = parsed.get('time_window')
        if tw == 'today':       date_filter = {'start': today, 'end': today}
        elif tw == 'yesterday': date_filter = {'start': today - timedelta(days=1), 'end': today - timedelta(days=1)}
        elif tw == 'this_week': date_filter = {'start': today - timedelta(days=today.weekday()), 'end': today}
        elif tw == 'last_week': date_filter = {'start': today - timedelta(days=today.weekday() + 7), 'end': today - timedelta(days=today.weekday() + 1)}
        elif tw == 'this_month': date_filter = {'start': today.replace(day=1), 'end': today}

        # ── Build queryset ────────────────────────────────────────────────
        qs = Employee.objects.select_related('site').all()
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                qs = qs.filter(site__in=profile.sites.all())
            except AdminProfile.DoesNotExist:
                qs = qs.none()

        if status:
            qs = qs.filter(status__iexact=status)
        if site:
            qs = qs.filter(site_id=site['id'])
        if position:
            qs = qs.filter(Q(position__iexact=position) | (Q(position__in=['', None]) & Q(salary_grade__iexact=position)))
        if department:
            qs = qs.filter(department__iexact=department)

        # Attendance-based filters
        if attendance_mod and date_filter:
            d_start, d_end = date_filter['start'], date_filter['end']
            att = Attendance.objects.filter(date__gte=d_start, date__lte=d_end)
            if attendance_mod == 'present':
                ids = att.filter(status='present').values_list('user_id', flat=True)
                qs = qs.filter(id__in=ids)
            elif attendance_mod == 'late':
                ids = att.filter(status='late').values_list('user_id', flat=True)
                qs = qs.filter(id__in=ids)
            elif attendance_mod == 'absent':
                # Active employees with no attendance record in the window.
                ids = att.values_list('user_id', flat=True).distinct()
                qs = qs.filter(status__iexact='Active').exclude(id__in=ids)

        # Avoid run-away lists
        sample_limit = 50
        results = list(qs.values('id', 'name', 'badge_number', 'department', 'position', 'site__name', 'status')[:sample_limit])
        total_count = qs.count()

        as_understood = {
            'intent':       intent,
            'status':       status,
            'site':         site['name'] if site else None,
            'position':     position,
            'department':   department,
            'attendance':   attendance_mod,
            'window':       (f"{date_filter['start']} → {date_filter['end']}"
                             if date_filter else None),
        }

        # Compose a friendly answer sentence
        bits = []
        if status:                              bits.append(f"on {status}")
        if attendance_mod:                       bits.append(attendance_mod)
        if position:                             bits.append(f'"{position}"')
        if department:                           bits.append(f'in {department}')
        if site:                                 bits.append(f'at {site["name"]}')
        if date_filter:                          bits.append(f"between {date_filter['start']} and {date_filter['end']}")
        descriptor = ' '.join(bits) or 'employees'
        if intent == 'count':
            answer = f"{total_count} {descriptor}".strip()
        else:
            if total_count == 0:
                answer = f"No {descriptor} found.".strip()
            else:
                answer = f"{total_count} {descriptor}".strip()
                if total_count > sample_limit:
                    answer += f" — showing first {sample_limit}"

        return Response({
            'question':     q_raw,
            'engine':       engine,                # 'llm' or 'heuristic'
            'as_understood': as_understood,
            'count':        total_count,
            'answer':       answer,
            'results':      [{
                'employee_id': r['id'],
                'name':        r['name'],
                'badge':       r['badge_number'] or '',
                'department':  r['department'] or '',
                'position':    r['position'] or '',
                'site':        r['site__name'] or '',
                'status':      r['status'] or '',
            } for r in results],
        })


# Template views
@login_required(login_url='admin-login')
def admin_manpower_recommendations_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'manpower_recommendations.html', {'is_superuser': request.user.is_superuser})


@login_required(login_url='admin-login')
def admin_ask_data_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'ask_data.html', {'is_superuser': request.user.is_superuser})


# ---------------------------------------------------------------------------
# Geofence smart-tuning
# ---------------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two WGS84 points."""
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    a = (radians(lat1), radians(lon1), radians(lat2), radians(lon2))
    d_lat = a[2] - a[0]
    d_lon = a[3] - a[1]
    h = sin(d_lat / 2) ** 2 + cos(a[0]) * cos(a[2]) * sin(d_lon / 2) ** 2
    return 2 * R * asin(sqrt(h))


def _point_in_polygon(lat, lon, polygon):
    """Ray-casting in lat/lon space. Polygon is [(lat, lon), ...]."""
    if not polygon or len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if ((lon_i > lon) != (lon_j > lon)) and \
           (lat < (lat_j - lat_i) * (lon - lon_i) / ((lon_j - lon_i) or 1e-12) + lat_i):
            inside = not inside
        j = i
    return inside


def _dist_to_segment_m(plat, plon, lat1, lon1, lat2, lon2):
    """Approximate point-to-segment distance in meters. Uses local equirect.
    projection — fine for sub-km segments where Earth curvature is negligible.
    """
    from math import radians, cos
    # Pick the segment midpoint's latitude for the projection scale
    mid_lat_rad = radians((lat1 + lat2) / 2)
    mx = 111320.0 * cos(mid_lat_rad)   # meters per degree of longitude at this latitude
    my = 110540.0                       # meters per degree of latitude (close enough)
    # Project both segment endpoints and the point into local x-y meters
    x1, y1 = (lon1 * mx, lat1 * my)
    x2, y2 = (lon2 * mx, lat2 * my)
    px, py = (plon * mx, plat * my)
    dx, dy = x2 - x1, y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len2))
    cx, cy = (x1 + t * dx, y1 + t * dy)
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _dist_to_polygon_m(lat, lon, polygon):
    """Min distance in meters from a point to any edge of the polygon."""
    if not polygon or len(polygon) < 2:
        return None
    best = float('inf')
    for i in range(len(polygon)):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % len(polygon)]
        d = _dist_to_segment_m(lat, lon, lat1, lon1, lat2, lon2)
        if d < best:
            best = d
    return best


def _cluster_points(points, eps_m=30.0):
    """Greedy single-link clustering — each new point joins the existing
    cluster whose centroid is within `eps_m` meters. Good enough at our scale.

    points: list of dicts {'lat': float, 'lon': float, ...meta}
    Returns: list of clusters; each = {'centroid': (lat, lon), 'points': [...]}.
    """
    clusters = []
    for p in points:
        joined = False
        for c in clusters:
            cl, cn = c['centroid']
            if _haversine_m(p['lat'], p['lon'], cl, cn) <= eps_m:
                c['points'].append(p)
                # Update centroid as running mean
                n = len(c['points'])
                avg_lat = (cl * (n - 1) + p['lat']) / n
                avg_lon = (cn * (n - 1) + p['lon']) / n
                c['centroid'] = (avg_lat, avg_lon)
                joined = True
                break
        if not joined:
            clusters.append({'centroid': (p['lat'], p['lon']), 'points': [p]})
    return clusters


def _polygon_centroid(polygon):
    if not polygon:
        return None
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def _direction_from(lat0, lon0, lat1, lon1):
    """Returns a compass direction (N/NE/E/...) from point 0 → point 1."""
    from math import atan2, degrees, radians, cos
    d_lon = (lon1 - lon0) * cos(radians((lat0 + lat1) / 2))
    d_lat = (lat1 - lat0)
    angle = (degrees(atan2(d_lon, d_lat)) + 360) % 360
    rose = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return rose[int((angle + 22.5) // 45) % 8]


class GeofenceTuningView(APIView):
    """GET /api/attendance/analytics/geofence-tuning/?site=&days=

    For each site (or just the requested one), pulls the last `days` of
    out-of-bounds attendance records, computes how far each miss was from the
    closest polygon edge, clusters them, and emits actionable suggestions:

    - "boundary": cluster is close to the polygon edge (likely a too-tight gate)
        → suggested action: extend the polygon ~Xm in compass direction Y
    - "off_site": cluster is far from the polygon (>200m)
        → suggested action: investigate; not a geofence issue
    - "near": single-digit hits very near the edge — quieter signal
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from datetime import timedelta

        try:
            days = max(1, min(int(request.GET.get('days') or 30), 365))
        except (TypeError, ValueError):
            days = 30

        site_filter = request.GET.get('site')
        sites_qs = Site.objects.exclude(coordinates__isnull=True).exclude(coordinates=[])
        if site_filter and site_filter != 'all':
            try:
                sites_qs = sites_qs.filter(id=int(site_filter))
            except (TypeError, ValueError):
                pass
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                sites_qs = sites_qs.filter(id__in=profile.sites.values_list('id', flat=True))
            except AdminProfile.DoesNotExist:
                sites_qs = sites_qs.none()

        today = timezone.localdate()
        window_start = today - timedelta(days=days)

        out = []
        for site in sites_qs:
            polygon = list(site.coordinates or [])
            if len(polygon) < 3:
                continue

            misses = list(Attendance.objects
                          .filter(user__site=site,
                                  is_within_geofence=False,
                                  date__gte=window_start,
                                  latitude__isnull=False, longitude__isnull=False)
                          .values('latitude', 'longitude', 'date', 'user_id'))
            if not misses:
                continue

            # Compute distance-to-edge for every miss
            points = []
            for m in misses:
                try:
                    lat = float(m['latitude']); lon = float(m['longitude'])
                except (TypeError, ValueError):
                    continue
                d_edge = _dist_to_polygon_m(lat, lon, polygon)
                if d_edge is None:
                    continue
                inside = _point_in_polygon(lat, lon, polygon)
                points.append({
                    'lat': lat, 'lon': lon,
                    'distance_m': d_edge,
                    'inside': inside,
                    'user_id': m['user_id'],
                    'date': m['date'],
                })

            if not points:
                continue

            clusters = _cluster_points(points, eps_m=30.0)
            poly_centroid = _polygon_centroid(polygon)

            cluster_recs = []
            for c in clusters:
                pts = c['points']
                n = len(pts)
                avg_dist = sum(p['distance_m'] for p in pts) / n
                unique_users = len({p['user_id'] for p in pts})
                # Categorise
                if avg_dist <= 50:
                    kind = 'boundary'
                    severity = 'high' if n >= 5 else ('med' if n >= 3 else 'low')
                elif avg_dist <= 200:
                    kind = 'near'
                    severity = 'med' if n >= 5 else 'low'
                else:
                    kind = 'off_site'
                    severity = 'low' if n < 5 else 'med'
                # Direction from polygon centroid out to cluster centroid
                cl, cn = c['centroid']
                direction = _direction_from(poly_centroid[0], poly_centroid[1], cl, cn) if poly_centroid else None

                suggestion = None
                if kind == 'boundary' and n >= 3:
                    add_m = int(round(min(avg_dist + 10, 50)))
                    suggestion = (f"Extend the geofence by ~{add_m} m on the {direction} side — "
                                  f"{n} miss{'es' if n != 1 else ''} from {unique_users} worker"
                                  f"{'s' if unique_users != 1 else ''} clustered there.")
                elif kind == 'off_site' and n >= 5:
                    suggestion = (f"Cluster sits {int(round(avg_dist))} m off-site ({direction}) — "
                                  f"likely workers at a different location, not a boundary issue.")

                cluster_recs.append({
                    'lat': cl, 'lon': cn,
                    'count': n,
                    'unique_users': unique_users,
                    'avg_distance_m': round(avg_dist, 1),
                    'direction': direction,
                    'kind': kind,
                    'severity': severity,
                    'suggestion': suggestion,
                })

            # Sort: boundary issues with most points first, then near, then off_site
            kind_rank = {'boundary': 0, 'near': 1, 'off_site': 2}
            cluster_recs.sort(key=lambda r: (kind_rank.get(r['kind'], 9), -r['count']))

            actionable = [c for c in cluster_recs if c.get('suggestion')]
            out.append({
                'site_id': site.id,
                'site_name': site.name,
                'polygon': [[lat, lon] for lat, lon in polygon],
                'centroid': list(poly_centroid) if poly_centroid else None,
                'total_misses': len(points),
                'clusters': cluster_recs,
                'has_action_items': bool(actionable),
                'action_summary': self._site_summary(cluster_recs, points),
            })

        # Sort: sites with most misses first
        out.sort(key=lambda s: -s['total_misses'])

        return Response({
            'as_of': str(today),
            'window_days': days,
            'sites': out,
        })

    @staticmethod
    def _site_summary(clusters, points):
        boundary = sum(1 for c in clusters if c['kind'] == 'boundary')
        off_site = sum(1 for c in clusters if c['kind'] == 'off_site')
        near_edge = sum(1 for p in points if p['distance_m'] <= 50)
        far = sum(1 for p in points if p['distance_m'] > 200)
        return {
            'total_misses': len(points),
            'near_edge_misses': near_edge,
            'far_misses': far,
            'boundary_clusters': boundary,
            'off_site_clusters': off_site,
        }


@login_required(login_url='admin-login')
def admin_geofence_tuning_view(request):
    if not request.user.is_staff:
        return redirect('admin-login')
    return render(request, 'geofence_tuning.html', {'is_superuser': request.user.is_superuser})


# ---------------------------------------------------------------------------
# Site Activity Report
# ---------------------------------------------------------------------------

class SiteActivityView(APIView):
    """GET /api/attendance/analytics/site-activity/?days=14

    Per-site activity report — which sites are actively recording attendance
    today, which are idle, and which haven't received any attendance in a
    while. Drives the dashboard's daily Site Activity panel.

    Per site:
      - active_employees:        total Active employees assigned to the site
      - present_today:           distinct employees with an Attendance row today
      - present_rate:            present_today / active_employees (percentage)
      - last_attendance:         most recent attendance datetime on this site
      - hours_since_last:        time since that attendance (None if never)
      - activity_status:         'active' | 'idle' | 'inactive' | 'never_used'
      - trend_7d:                attendance counts for the last 7 days (oldest → newest)
      - avg_7d:                  7-day moving average
      - delta_vs_avg:            today vs 7-day average (percentage points)

    Activity bands (tunable via ?idle_hours= / ?inactive_days=):
      active     — at least one attendance in the last `idle_hours` (default 24h)
      idle       — last attendance 1-`inactive_days` days ago
      inactive   — last attendance > `inactive_days` days ago (default 7d)
      never_used — no attendance recorded ever
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        from datetime import timedelta
        from collections import defaultdict
        from django.db.models import Max, Count

        try:
            days = max(1, min(int(request.GET.get('days') or 14), 90))
        except (TypeError, ValueError):
            days = 14
        try:
            idle_hours = max(1, min(int(request.GET.get('idle_hours') or 24), 168))
        except (TypeError, ValueError):
            idle_hours = 24
        try:
            inactive_days = max(1, min(int(request.GET.get('inactive_days') or 7), 90))
        except (TypeError, ValueError):
            inactive_days = 7

        now = timezone.now()
        today = timezone.localdate()

        # ── Scope sites by site-admin permission ──────────────────────────
        sites_qs = Site.objects.all().order_by('name')
        if not request.user.is_superuser:
            try:
                profile = AdminProfile.objects.get(user=request.user)
                sites_qs = sites_qs.filter(id__in=profile.sites.values_list('id', flat=True))
            except AdminProfile.DoesNotExist:
                sites_qs = sites_qs.none()
        site_ids = list(sites_qs.values_list('id', flat=True))
        if not site_ids:
            return Response({'as_of': str(today), 'sites': [],
                             'totals': {}, 'inactive_sites': []})

        # ── Active-employee counts per site ───────────────────────────────
        emp_counts = dict(
            Employee.objects
            .filter(status__iexact='Active', site_id__in=site_ids)
            .values('site_id')
            .annotate(n=Count('id'))
            .values_list('site_id', 'n')
        )

        # ── Today's distinct attended employees per site ──────────────────
        present_today = dict(
            Attendance.objects
            .filter(date=today, user__site_id__in=site_ids)
            .values('user__site_id')
            .annotate(n=Count('user_id', distinct=True))
            .values_list('user__site_id', 'n')
        )

        # ── Most recent attendance per site (by check_in_time) ────────────
        last_attendance_raw = dict(
            Attendance.objects
            .filter(user__site_id__in=site_ids, check_in_time__isnull=False)
            .values('user__site_id')
            .annotate(latest=Max('check_in_time'))
            .values_list('user__site_id', 'latest')
        )

        # ── 7-day rolling attendance counts per site (for trend chart) ───
        trend_start = today - timedelta(days=6)
        trend_rows = (
            Attendance.objects
            .filter(date__gte=trend_start, date__lte=today, user__site_id__in=site_ids)
            .values('user__site_id', 'date')
            .annotate(n=Count('user_id', distinct=True))
            .values_list('user__site_id', 'date', 'n')
        )
        per_site_per_day = defaultdict(lambda: defaultdict(int))
        for sid, d, n in trend_rows:
            per_site_per_day[sid][d] = n

        # ── Compose per-site rows ─────────────────────────────────────────
        rows = []
        for site in sites_qs:
            sid = site.id
            active_emps = emp_counts.get(sid, 0)
            today_count = present_today.get(sid, 0)
            last_ci = last_attendance_raw.get(sid)
            hours_since = None
            if last_ci:
                hours_since = round((now - last_ci).total_seconds() / 3600.0, 1)

            if last_ci is None:
                status_band = 'never_used'
            elif hours_since is not None and hours_since <= idle_hours:
                status_band = 'active'
            elif hours_since is not None and hours_since <= inactive_days * 24:
                status_band = 'idle'
            else:
                status_band = 'inactive'

            # Trend array: 7 days oldest → newest
            trend = []
            for offset in range(6, -1, -1):
                d = today - timedelta(days=offset)
                trend.append(per_site_per_day[sid].get(d, 0))
            avg7 = round(sum(trend) / 7.0, 1) if trend else 0
            delta = round(today_count - avg7, 1)

            present_rate = round(today_count / active_emps * 100, 1) if active_emps else 0.0

            rows.append({
                'site_id':           sid,
                'site_name':         site.name,
                'active_employees':  active_emps,
                'present_today':     today_count,
                'present_rate':      present_rate,
                'last_attendance':   last_ci.isoformat() if last_ci else None,
                'hours_since_last':  hours_since,
                'activity_status':   status_band,
                'trend_7d':          trend,
                'avg_7d':            avg7,
                'delta_vs_avg':      delta,
            })

        # Sort: inactive + never_used first (admin needs to act on these),
        # then by present_today desc so busy sites surface near the top.
        order = {'never_used': 0, 'inactive': 1, 'idle': 2, 'active': 3}
        rows.sort(key=lambda r: (order.get(r['activity_status'], 9), -r['present_today']))

        # ── Roll-up totals ────────────────────────────────────────────────
        totals = {
            'sites_total':      len(rows),
            'sites_active':     sum(1 for r in rows if r['activity_status'] == 'active'),
            'sites_idle':       sum(1 for r in rows if r['activity_status'] == 'idle'),
            'sites_inactive':   sum(1 for r in rows if r['activity_status'] == 'inactive'),
            'sites_never_used': sum(1 for r in rows if r['activity_status'] == 'never_used'),
            'present_today':    sum(r['present_today'] for r in rows),
            'active_employees': sum(r['active_employees'] for r in rows),
        }
        totals['overall_present_rate'] = (
            round(totals['present_today'] / totals['active_employees'] * 100, 1)
            if totals['active_employees'] else 0.0
        )

        # Quick-access list of the "system not used" sites for the alert callout
        inactive_sites = [
            {'site_id': r['site_id'], 'site_name': r['site_name'],
             'hours_since_last': r['hours_since_last'],
             'activity_status': r['activity_status'],
             'active_employees': r['active_employees']}
            for r in rows
            if r['activity_status'] in ('inactive', 'never_used')
        ]

        return Response({
            'as_of':           str(today),
            'idle_hours':      idle_hours,
            'inactive_days':   inactive_days,
            'totals':          totals,
            'sites':           rows,
            'inactive_sites':  inactive_sites,
        })


# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS — global app configuration (superuser only)
# ══════════════════════════════════════════════════════════════════════════
from datetime import datetime as _dt


def _fmt_time(t):
    """Serialize a TimeField value to 'HH:MM' for the settings form."""
    if not t:
        return ''
    try:
        return t.strftime('%H:%M')
    except Exception:
        return str(t)[:5]


def _parse_time(val, fallback=None):
    """Parse 'HH:MM' or 'HH:MM:SS' into a time, else return fallback."""
    if not val:
        return fallback
    for fmt in ('%H:%M', '%H:%M:%S'):
        try:
            return _dt.strptime(str(val).strip(), fmt).time()
        except (ValueError, TypeError):
            continue
    return fallback


def _settings_to_dict(s):
    """Full JSON representation of AppSettings for the settings page + dashboard."""
    return {
        # Attendance & shift rules
        'default_office_start_time': _fmt_time(s.default_office_start_time),
        'default_office_end_time':   _fmt_time(s.default_office_end_time),
        'default_worker_start_time': _fmt_time(s.default_worker_start_time),
        'default_worker_end_time':   _fmt_time(s.default_worker_end_time),
        'default_office_day_off':    s.default_office_day_off or '',
        'default_worker_day_off':    s.default_worker_day_off or '',
        'late_grace_minutes':        s.late_grace_minutes,
        'half_day_threshold_hours':  float(s.half_day_threshold_hours or 0),
        'normal_ot_threshold_minutes': s.normal_ot_threshold_minutes,
        'weekend_days':              s.weekend_days or [],
        # Geofence / location
        'default_geofence_radius_meters': s.default_geofence_radius_meters,
        'gps_accuracy_tolerance_meters':  s.gps_accuracy_tolerance_meters,
        # Master lists
        'sponsors':  s.sponsors or [],
        'employers': s.employers or [],
        # Salary
        'currency_code':         s.currency_code or 'AED',
        'salary_superuser_only': s.salary_superuser_only,
        'gross_formula':         s.gross_formula or {},
        # Documents & compliance
        'passport_reminder_lead_days':    s.passport_reminder_lead_days,
        'visa_reminder_lead_days':        s.visa_reminder_lead_days,
        'labour_card_reminder_lead_days': s.labour_card_reminder_lead_days,
        'mol_reminder_lead_days':         s.mol_reminder_lead_days,
        'expiry_alert_recipients':        s.expiry_alert_recipients or [],
        # Data retention
        'distribution_snapshot_retention_months': s.distribution_snapshot_retention_months,
        # Navigation
        'nav_visibility': s.nav_visibility or {},
        'updated_at': str(s.updated_at) if s.updated_at else '',
    }


@login_required(login_url="admin-login")
def admin_settings_view(request):
    """Settings page — superuser only."""
    import json
    if not request.user.is_superuser:
        return redirect("admin-dashboard")
    s = AppSettings.load()
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return render(request, "settings.html", {
        'is_superuser': True,
        'settings_json': json.dumps(_settings_to_dict(s)),
        'sites': Site.objects.all().order_by('name'),
        'days_of_week': DAYS,
        'nav_links': [
            ('dashboard', 'Dashboard'), ('reports', 'Reports'),
            ('monthly_report', 'Monthly Report'), ('distribution_list', 'Distribution List'),
            ('departments', 'Departments'), ('user_face', 'User Face'),
            ('attrition_risk', 'Attrition Risk'), ('document_expiry', 'Document Expiry'),
            ('manpower_recs', 'Manpower Recs'), ('ask_data', 'Ask the Data'),
            ('geofence_tuning', 'Geofence Tuning'), ('salary', 'Salary'),
            ('sites', 'Sites'), ('site_admins', 'Site Admins'), ('settings', 'Settings'),
        ],
    })


class AppSettingsView(APIView):
    """GET / PUT the global AppSettings singleton (superuser only)."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        if not request.user.is_superuser:
            return Response({'error': 'Permission denied'}, status=403)
        return Response(_settings_to_dict(AppSettings.load()))

    def put(self, request):
        if not request.user.is_superuser:
            return Response({'error': 'Permission denied'}, status=403)
        s = AppSettings.load()
        d = request.data

        # --- Times ---
        for f in ('default_office_start_time', 'default_office_end_time',
                  'default_worker_start_time', 'default_worker_end_time'):
            if f in d:
                parsed = _parse_time(d.get(f), getattr(s, f))
                if parsed is not None:
                    setattr(s, f, parsed)

        # --- Plain char / day-off ---
        for f in ('default_office_day_off', 'default_worker_day_off', 'currency_code'):
            if f in d:
                setattr(s, f, (d.get(f) or '').strip())

        # --- Non-negative integers ---
        for f in ('late_grace_minutes', 'normal_ot_threshold_minutes',
                  'passport_reminder_lead_days', 'visa_reminder_lead_days',
                  'labour_card_reminder_lead_days', 'mol_reminder_lead_days',
                  'distribution_snapshot_retention_months'):
            if f in d:
                try:
                    setattr(s, f, max(0, int(d.get(f))))
                except (ValueError, TypeError):
                    pass

        # --- Floats ---
        for f in ('default_geofence_radius_meters', 'gps_accuracy_tolerance_meters'):
            if f in d:
                try:
                    setattr(s, f, max(0.0, float(d.get(f))))
                except (ValueError, TypeError):
                    pass

        if 'half_day_threshold_hours' in d:
            try:
                s.half_day_threshold_hours = max(0, float(d.get('half_day_threshold_hours')))
            except (ValueError, TypeError):
                pass

        # --- Booleans ---
        if 'salary_superuser_only' in d:
            s.salary_superuser_only = bool(d.get('salary_superuser_only'))

        # --- Lists (sponsors / employers / recipients / weekend days) ---
        def _clean_list(val):
            if not isinstance(val, list):
                return None
            seen, out = set(), []
            for x in val:
                x = (str(x) or '').strip()
                if x and x.lower() not in seen:
                    seen.add(x.lower())
                    out.append(x)
            return out

        for f in ('sponsors', 'employers', 'expiry_alert_recipients', 'weekend_days'):
            if f in d:
                cleaned = _clean_list(d.get(f))
                if cleaned is not None:
                    setattr(s, f, cleaned)

        # --- JSON dicts (gross formula, nav visibility) ---
        if 'gross_formula' in d and isinstance(d.get('gross_formula'), dict):
            gf = {}
            for k, v in d['gross_formula'].items():
                try:
                    gf[k] = 1 if int(v) > 0 else (-1 if int(v) < 0 else 0)
                except (ValueError, TypeError):
                    gf[k] = 0
            s.gross_formula = gf

        if 'nav_visibility' in d and isinstance(d.get('nav_visibility'), dict):
            s.nav_visibility = {k: bool(v) for k, v in d['nav_visibility'].items()}

        s.updated_by = request.user
        s.save()
        return Response({'success': True, 'settings': _settings_to_dict(s)})


class PublicHolidayView(APIView):
    """List / create / delete org-wide public holidays (superuser only)."""
    permission_classes = [IsAdminUser]

    def get(self, request):
        rows = PublicHoliday.objects.all()
        return Response({'holidays': [
            {'id': h.id, 'name': h.name, 'date': str(h.date),
             'recurring_annually': h.recurring_annually}
            for h in rows
        ]})

    def post(self, request):
        if not request.user.is_superuser:
            return Response({'error': 'Permission denied'}, status=403)
        name = (request.data.get('name') or '').strip()
        date = (request.data.get('date') or '').strip()
        if not name or not date:
            return Response({'error': 'Name and date are required.'}, status=400)
        try:
            parsed = _dt.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            return Response({'error': 'Invalid date (use YYYY-MM-DD).'}, status=400)
        h = PublicHoliday.objects.create(
            name=name, date=parsed,
            recurring_annually=bool(request.data.get('recurring_annually')),
        )
        return Response({'success': True, 'id': h.id,
                         'name': h.name, 'date': str(h.date),
                         'recurring_annually': h.recurring_annually}, status=201)

    def delete(self, request, holiday_id=None):
        if not request.user.is_superuser:
            return Response({'error': 'Permission denied'}, status=403)
        PublicHoliday.objects.filter(id=holiday_id).delete()
        return Response({'success': True})


# ══════════════════════════════════════════════════════════════════════════
#  MISSING CHECK-IN — checked out without checking in (notify super admin)
# ══════════════════════════════════════════════════════════════════════════
def _missing_checkin_qs(request):
    """Attendance rows where the employee checked OUT but never checked IN.

    Honours site-admin scoping plus optional ?site=<id|all> and ?date=YYYY-MM-DD
    filters. No date → last 30 days. Shared by the list + export endpoints.
    """
    from datetime import datetime as _dt, timedelta
    qs = Attendance.objects.select_related('user', 'user__site').filter(
        check_out_time__isnull=False, check_in_time__isnull=True,
    )
    if not request.user.is_superuser:
        try:
            profile = AdminProfile.objects.get(user=request.user)
            qs = qs.filter(user__site__in=profile.sites.all())
        except AdminProfile.DoesNotExist:
            qs = qs.none()

    # Site filter — mirrors the dashboard's site selector.
    site_id = (request.GET.get('site') or '').strip()
    if site_id and site_id.lower() != 'all':
        try:
            qs = qs.filter(user__site_id=int(site_id))
        except (TypeError, ValueError):
            pass

    date_str = (request.GET.get('date') or '').strip()
    if date_str:
        try:
            qs = qs.filter(date=_dt.strptime(date_str, '%Y-%m-%d').date())
        except ValueError:
            pass
    else:
        qs = qs.filter(date__gte=timezone.localdate() - timedelta(days=30))

    return qs.order_by('-date', 'user__name')


class MissingCheckInView(APIView):
    """List attendance rows where the person checked OUT but never checked IN.

    GET /api/attendance/missing-checkin/?site=<id|all>&date=YYYY-MM-DD
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        qs = _missing_checkin_qs(request)
        total = qs.count()
        rows = [{
            'id': a.id,
            'employee': a.user.name,
            'employee_id': a.user.id,
            'badge_number': a.user.badge_number or '-',
            'site': a.user.site.name if a.user.site else '-',
            'date': str(a.date),
            'check_out': timezone.localtime(a.check_out_time).strftime('%I:%M %p') if a.check_out_time else '-',
        } for a in qs[:300]]
        # `total` is the unclipped count so the UI can say "showing 300 of N".
        return Response({'count': len(rows), 'total': total, 'rows': rows})


class MissingCheckInExportView(APIView):
    """Excel download of the missing check-ins for the current site/date filter."""
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request):
        import io
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from django.http import HttpResponse

        qs = _missing_checkin_qs(request)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Missing Check-In'

        headers = ['Badge ID', 'Employee', 'Site', 'Date', 'Checked Out']
        ws.append(headers)
        fill = PatternFill(start_color='2563EB', end_color='2563EB', fill_type='solid')
        font = Font(bold=True, color='FFFFFF')
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(horizontal='center')
            ws.column_dimensions[cell.column_letter].width = 24
        ws.freeze_panes = 'A2'

        for a in qs:
            ws.append([
                a.user.badge_number or '-',
                a.user.name,
                a.user.site.name if a.user.site else '-',
                str(a.date),
                timezone.localtime(a.check_out_time).strftime('%I:%M %p') if a.check_out_time else '-',
            ])

        out = io.BytesIO()
        wb.save(out)
        out.seek(0)
        resp = HttpResponse(
            out.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        resp['Content-Disposition'] = 'attachment; filename=missing_check_in.xlsx'
        return resp


class SetCheckInView(APIView):
    """Super-admin fills in a missing check-in time for one attendance row.

    POST /api/attendance/attendance/<id>/set-checkin/  body: {"time": "HH:MM"}
    """
    permission_classes = [IsAdminUser]

    def post(self, request, attendance_id):
        if not request.user.is_superuser:
            return Response({'error': 'Permission denied. Super admin only.'}, status=403)

        from datetime import datetime as _dt
        time_str = (request.data.get('time') or '').strip()
        if not time_str:
            return Response({'error': 'Provide a check-in time (HH:MM).'}, status=400)
        parsed = None
        for fmt in ('%H:%M', '%H:%M:%S', '%I:%M %p'):
            try:
                parsed = _dt.strptime(time_str, fmt).time()
                break
            except ValueError:
                continue
        if parsed is None:
            return Response({'error': 'Invalid time. Use HH:MM (24-hour).'}, status=400)

        try:
            a = Attendance.objects.select_related('user').get(id=attendance_id)
        except Attendance.DoesNotExist:
            return Response({'error': 'Attendance record not found.'}, status=404)

        aware = timezone.make_aware(
            _dt.combine(a.date, parsed), timezone.get_current_timezone(),
        )
        if a.check_out_time and aware > a.check_out_time:
            return Response(
                {'error': 'Check-in time cannot be after the check-out time.'}, status=400,
            )
        a.check_in_time = aware
        try:
            a.calculate_late_and_early()
        except Exception:  # noqa: BLE001
            logger.exception('recompute after set-checkin failed for attendance=%s', a.id)
        a.save()
        return Response({
            'success': True,
            'employee': a.user.name,
            'check_in': timezone.localtime(a.check_in_time).strftime('%I:%M %p'),
            'late_minutes': a.late_minutes,
        })


# ══════════════════════════════════════════════════════════════════════════
#  EMPLOYEE GENERAL NOTES — multiple dated notes per employee
# ══════════════════════════════════════════════════════════════════════════
class EmployeeGeneralNotesView(APIView):
    """List / add / delete general notes for an employee.

    GET    /api/attendance/employees/<id>/general-notes/            → list
    POST   /api/attendance/employees/<id>/general-notes/            → add: subject, date, note
    DELETE /api/attendance/employees/<id>/general-notes/?id=<note>  → remove one
    """
    permission_classes = [IsAdminUser | IsSiteAdmin]

    def get(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        rows = emp.general_notes.select_related('created_by').all()
        items = [{
            'id': n.id,
            'subject': n.subject or '',
            'date': str(n.date) if n.date else '',
            'note': n.note or '',
            'created_at': n.created_at.isoformat() if n.created_at else None,
            'created_by': n.created_by.username if n.created_by else None,
        } for n in rows]
        return Response({'employee_id': emp.id, 'count': len(items), 'notes': items})

    def post(self, request, employee_id):
        try:
            emp = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=404)
        subject = (request.data.get('subject') or '').strip()
        note = (request.data.get('note') or '').strip()
        date_raw = (request.data.get('date') or '').strip()
        if not subject and not note:
            return Response({'error': 'Enter a subject or a note.'}, status=400)
        from datetime import datetime as _dt
        d = None
        if date_raw:
            for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
                try:
                    d = _dt.strptime(date_raw, fmt).date()
                    break
                except ValueError:
                    continue
        n = EmployeeGeneralNote.objects.create(
            employee=emp,
            subject=(subject[:200] or None),
            date=d,
            note=(note or None),
            created_by=request.user if request.user.is_authenticated else None,
        )
        return Response({
            'success': True, 'id': n.id,
            'subject': n.subject or '', 'date': str(n.date) if n.date else '',
            'note': n.note or '', 'created_at': n.created_at.isoformat(),
            'created_by': n.created_by.username if n.created_by else None,
        }, status=201)

    def delete(self, request, employee_id):
        note_id = request.query_params.get('id') or request.data.get('id')
        if not note_id:
            return Response({'error': 'Note id is required.'}, status=400)
        try:
            n = EmployeeGeneralNote.objects.get(id=note_id, employee_id=employee_id)
        except EmployeeGeneralNote.DoesNotExist:
            return Response({'error': 'Note not found'}, status=404)
        n.delete()
        return Response({'success': True})
