"""
Signal handlers for hr_manpower.

Events emitted (logged + extensible for external event bus):
  - hr.employee.onboarded        → when EmployeeHRProfile is created
  - hr.employee.assigned         → when ProjectAssignment status → Active
  - hr.employee.offboarded       → handled in OffboardingView directly
  - attendance.approved          → when AttendanceExtension approval_status → Approved
  - labor_cost.posted            → when ActualCost record is created

Automation:
  - attendance.approved → auto-post ActualCost (zero manual journals)
  - ProjectAssignment released → warn if no camp released
"""
import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import (
    EmployeeHRProfile,
    ProjectAssignment,
    AttendanceExtension,
    ActualCost,
    CampAllocation,
)

logger = logging.getLogger('hr_manpower')


# ---------------------------------------------------------------------------
# HELPER: post labor cost after attendance approval
# ---------------------------------------------------------------------------

def post_labor_cost(ext: AttendanceExtension):
    """
    Create an ActualCost record from an approved AttendanceExtension.
    Idempotent: skips if already posted.
    """
    if ext.cost_posted:
        return
    if not ext.project:
        logger.warning(
            'AttendanceExtension %s approved but has no project — skipping cost post.', ext.id
        )
        return

    employee = ext.attendance.user
    attendance = ext.attendance

    # Get rates from HR Profile (fallback to 0 if no profile)
    basic_rate = 0
    ot_rate_multiplier = 1.5
    cbs_code = 'LABOR-GENERAL'

    try:
        hr_profile = employee.hr_profile
        basic_rate = float(hr_profile.basic_rate or 0)
        ot_rate_multiplier = float(hr_profile.overtime_rate_multiplier or 1.5)
        cbs_code = hr_profile.cost_center_default or 'LABOR-GENERAL'
    except EmployeeHRProfile.DoesNotExist:
        logger.warning(
            'Employee %s has no HR Profile. Using zero rates for cost posting.', employee.id
        )

    regular_hours = float(ext.total_hours or 0)
    overtime_hours = float(attendance.normal_ot_hours or 0) + float(attendance.special_ot_hours or 0)
    ot_rate = basic_rate * ot_rate_multiplier

    cost = ActualCost.objects.create(
        project=ext.project,
        activity=ext.activity,
        employee=employee,
        attendance_ext=ext,
        cbs_code=cbs_code,
        cost_date=attendance.date,
        regular_hours=regular_hours,
        overtime_hours=overtime_hours,
        basic_rate=basic_rate,
        overtime_rate=ot_rate,
        # regular_cost, overtime_cost, total_cost are auto-calculated in ActualCost.save()
    )

    ext.cost_posted = True
    ext.save(update_fields=['cost_posted'])

    logger.info(
        'EVENT attendance.approved → labor_cost.posted | employee=%s project=%s date=%s total_cost=%s',
        employee.name, ext.project.project_code, attendance.date, cost.total_cost,
    )


# ---------------------------------------------------------------------------
# SIGNAL: AttendanceExtension saved → trigger cost posting when approved
# ---------------------------------------------------------------------------

@receiver(post_save, sender=AttendanceExtension)
def on_attendance_extension_saved(sender, instance, created, **kwargs):
    if instance.approval_status == 'Approved' and not instance.cost_posted:
        post_labor_cost(instance)


# ---------------------------------------------------------------------------
# SIGNAL: EmployeeHRProfile created → emit hr.employee.onboarded
# ---------------------------------------------------------------------------

@receiver(post_save, sender=EmployeeHRProfile)
def on_hr_profile_created(sender, instance, created, **kwargs):
    if created:
        logger.info(
            'EVENT hr.employee.onboarded | employee=%s code=%s',
            instance.employee.name, instance.employee_code,
        )


# ---------------------------------------------------------------------------
# SIGNAL: ProjectAssignment saved → emit hr.employee.assigned on activation
# ---------------------------------------------------------------------------

@receiver(post_save, sender=ProjectAssignment)
def on_project_assignment_saved(sender, instance, created, **kwargs):
    if instance.status == 'Active':
        logger.info(
            'EVENT hr.employee.assigned | employee=%s project=%s activity=%s',
            instance.employee.name,
            instance.project.project_code,
            instance.activity.activity_name if instance.activity else 'N/A',
        )

    if instance.status == 'Released':
        # Check if camp bed is still active — warn if so
        active_camp = CampAllocation.objects.filter(
            employee=instance.employee, status='Active'
        ).first()
        if active_camp:
            logger.warning(
                'WARN hr.employee.released_from_project but camp bed still active | '
                'employee=%s camp=%s room=%s bed=%s',
                instance.employee.name,
                active_camp.room.camp.name,
                active_camp.room.room_number,
                active_camp.bed_number,
            )
