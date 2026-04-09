from django.db import transaction
from django.db.models import Sum, Count, Q
from django.utils import timezone

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from attendance.models import Employee, Attendance

from .models import (
    Trade, EmployeeHRProfile, Project, Activity,
    ManpowerDemand, ProjectAssignment,
    AttendanceExtension, ActualCost,
    ProductivityRecord, Camp, CampRoom, CampAllocation,
)
from .serializers import (
    TradeSerializer, EmployeeHRProfileSerializer, EmployeeWithHRProfileSerializer,
    ProjectSerializer, ProjectListSerializer, ActivitySerializer,
    ManpowerDemandSerializer, ProjectAssignmentSerializer,
    AttendanceExtensionSerializer, AttendanceExtensionBulkCreateSerializer,
    ActualCostSerializer, ProductivityRecordSerializer,
    CampSerializer, CampListSerializer, CampRoomSerializer, CampAllocationSerializer,
    DemandVsSupplySerializer, CampOccupancySerializer,
)
from .permissions import (
    IsHRAdmin, IsHROfficer, IsProjectManager, IsForeman, IsFinance, IsHRAdminOrReadOnly,
)


# ---------------------------------------------------------------------------
# TRADE
# ---------------------------------------------------------------------------

class TradeViewSet(viewsets.ModelViewSet):
    queryset = Trade.objects.all()
    serializer_class = TradeSerializer
    permission_classes = [IsHRAdminOrReadOnly]

    def get_queryset(self):
        qs = Trade.objects.all()
        category = self.request.query_params.get('category')
        if category:
            qs = qs.filter(category=category)
        return qs


# ---------------------------------------------------------------------------
# EMPLOYEE HR PROFILE
# ---------------------------------------------------------------------------

class EmployeeHRProfileViewSet(viewsets.ModelViewSet):
    queryset = EmployeeHRProfile.objects.select_related('employee', 'trade').all()
    serializer_class = EmployeeHRProfileSerializer
    permission_classes = [IsHROfficer]

    def get_queryset(self):
        qs = EmployeeHRProfile.objects.select_related('employee', 'trade').all()
        trade_id = self.request.query_params.get('trade_id')
        skill_level = self.request.query_params.get('skill_level')
        employment_type = self.request.query_params.get('employment_type')
        docs_valid = self.request.query_params.get('documents_valid')

        if trade_id:
            qs = qs.filter(trade_id=trade_id)
        if skill_level:
            qs = qs.filter(skill_level=skill_level)
        if employment_type:
            qs = qs.filter(employment_type=employment_type)
        if docs_valid is not None:
            qs = qs.filter(documents_valid=docs_valid.lower() == 'true')
        return qs

    @action(detail=True, methods=['post'], permission_classes=[IsHRAdmin])
    def validate_documents(self, request, pk=None):
        """Mark employee documents as valid."""
        profile = self.get_object()
        profile.documents_valid = True
        profile.save(update_fields=['documents_valid', 'updated_at'])
        return Response({'status': 'documents marked as valid'})

    @action(detail=True, methods=['post'], permission_classes=[IsHRAdmin])
    def invalidate_documents(self, request, pk=None):
        """Mark employee documents as invalid (e.g. visa expired)."""
        profile = self.get_object()
        profile.documents_valid = False
        profile.save(update_fields=['documents_valid', 'updated_at'])
        return Response({'status': 'documents marked as invalid'})


class EmployeeWithHRProfileView(APIView):
    """Read-only: list all employees enriched with HR Profile data."""
    permission_classes = [IsHROfficer]

    def get(self, request):
        employees = Employee.objects.select_related('hr_profile__trade', 'site').all()
        name = request.query_params.get('name')
        status_filter = request.query_params.get('status')
        trade_id = request.query_params.get('trade_id')

        if name:
            employees = employees.filter(name__icontains=name)
        if status_filter:
            employees = employees.filter(status=status_filter)
        if trade_id:
            employees = employees.filter(hr_profile__trade_id=trade_id)

        serializer = EmployeeWithHRProfileSerializer(employees, many=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------------

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.select_related('site').all()
    permission_classes = [IsHRAdminOrReadOnly]

    def get_serializer_class(self):
        if self.action == 'list':
            return ProjectListSerializer
        return ProjectSerializer

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        qs = Project.objects.select_related('site').all()
        status_filter = self.request.query_params.get('status')
        site_id = self.request.query_params.get('site_id')
        if status_filter:
            qs = qs.filter(status=status_filter)
        if site_id:
            qs = qs.filter(site_id=site_id)
        return qs

    @action(detail=True, methods=['get'])
    def demand_vs_supply(self, request, pk=None):
        """Dashboard: manpower demand vs current supply for a project."""
        project = self.get_object()
        demands = ManpowerDemand.objects.filter(project=project).select_related('trade')
        result = []
        for d in demands:
            assigned = ProjectAssignment.objects.filter(
                project=project,
                employee__hr_profile__trade=d.trade,
                status__in=['Planned', 'Active'],
                start_date__lte=d.required_end_date,
            ).filter(
                Q(end_date__isnull=True) | Q(end_date__gte=d.required_start_date)
            ).count()
            result.append({
                'trade_id': d.trade.id,
                'trade_name': d.trade.trade_name,
                'required_qty': d.required_qty,
                'assigned_qty': assigned,
                'gap': d.required_qty - assigned,
                'demand_start': d.required_start_date,
                'demand_end': d.required_end_date,
            })
        serializer = DemandVsSupplySerializer(result, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def labor_cost_summary(self, request, pk=None):
        """Finance: total labor cost breakdown for a project."""
        project = self.get_object()
        summary = ActualCost.objects.filter(project=project).aggregate(
            total_regular_hours=Sum('regular_hours'),
            total_overtime_hours=Sum('overtime_hours'),
            total_regular_cost=Sum('regular_cost'),
            total_overtime_cost=Sum('overtime_cost'),
            total_cost=Sum('total_cost'),
            total_records=Count('id'),
        )
        return Response(summary)


# ---------------------------------------------------------------------------
# ACTIVITY
# ---------------------------------------------------------------------------

class ActivityViewSet(viewsets.ModelViewSet):
    queryset = Activity.objects.select_related('project', 'trade').all()
    serializer_class = ActivitySerializer
    permission_classes = [IsHRAdminOrReadOnly]

    def get_queryset(self):
        qs = Activity.objects.select_related('project', 'trade').all()
        project_id = self.request.query_params.get('project_id')
        trade_id = self.request.query_params.get('trade_id')
        status_filter = self.request.query_params.get('status')
        if project_id:
            qs = qs.filter(project_id=project_id)
        if trade_id:
            qs = qs.filter(trade_id=trade_id)
        if status_filter:
            qs = qs.filter(status=status_filter)
        return qs


# ---------------------------------------------------------------------------
# MANPOWER DEMAND
# ---------------------------------------------------------------------------

class ManpowerDemandViewSet(viewsets.ModelViewSet):
    queryset = ManpowerDemand.objects.select_related('project', 'activity', 'trade').all()
    serializer_class = ManpowerDemandSerializer
    permission_classes = [IsProjectManager]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def get_queryset(self):
        qs = ManpowerDemand.objects.select_related('project', 'activity', 'trade').all()
        project_id = self.request.query_params.get('project_id')
        trade_id = self.request.query_params.get('trade_id')
        source = self.request.query_params.get('source')
        if project_id:
            qs = qs.filter(project_id=project_id)
        if trade_id:
            qs = qs.filter(trade_id=trade_id)
        if source:
            qs = qs.filter(source=source)
        return qs

    @action(detail=False, methods=['post'], permission_classes=[IsProjectManager])
    def generate_from_activity(self, request):
        """
        Simulate Planning → Demand generation.
        Accepts a list of activity IDs and creates demand records from Activity data.
        Body: { "activity_ids": [1, 2, 3] }
        """
        activity_ids = request.data.get('activity_ids', [])
        if not activity_ids:
            return Response({'error': 'activity_ids is required'}, status=status.HTTP_400_BAD_REQUEST)

        activities = Activity.objects.filter(id__in=activity_ids).select_related('project', 'trade')
        created = []
        for act in activities:
            if not act.trade:
                continue
            days = max((act.end_date - act.start_date).days + 1, 1)
            planned_hours = 8 * days  # default crew of 1 × 8h/day; HR adjusts qty
            demand, _ = ManpowerDemand.objects.get_or_create(
                project=act.project,
                activity=act,
                trade=act.trade,
                defaults={
                    'required_qty': 1,
                    'required_start_date': act.start_date,
                    'required_end_date': act.end_date,
                    'planned_man_hours': planned_hours,
                    'source': 'planning',
                    'created_by': request.user,
                }
            )
            created.append(demand.id)

        return Response({
            'message': f'{len(created)} demand record(s) created/found.',
            'demand_ids': created,
        }, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# PROJECT ASSIGNMENT (Allocation)
# ---------------------------------------------------------------------------

class ProjectAssignmentViewSet(viewsets.ModelViewSet):
    queryset = ProjectAssignment.objects.select_related('employee', 'project', 'activity').all()
    serializer_class = ProjectAssignmentSerializer
    permission_classes = [IsProjectManager]

    def perform_create(self, serializer):
        assignment = serializer.save(assigned_by=self.request.user)
        # Emit event (handled by signal)
        return assignment

    def get_queryset(self):
        qs = ProjectAssignment.objects.select_related('employee', 'project', 'activity').all()
        project_id = self.request.query_params.get('project_id')
        employee_id = self.request.query_params.get('employee_id')
        assignment_status = self.request.query_params.get('status')
        if project_id:
            qs = qs.filter(project_id=project_id)
        if employee_id:
            qs = qs.filter(employee_id=employee_id)
        if assignment_status:
            qs = qs.filter(status=assignment_status)
        return qs

    @action(detail=True, methods=['post'], permission_classes=[IsProjectManager])
    def activate(self, request, pk=None):
        """Move assignment from Planned → Active."""
        assignment = self.get_object()
        if assignment.status != 'Planned':
            return Response(
                {'error': f'Cannot activate. Current status: {assignment.status}'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        assignment.status = 'Active'
        assignment.save(update_fields=['status', 'updated_at'])
        return Response(ProjectAssignmentSerializer(assignment).data)

    @action(detail=True, methods=['post'], permission_classes=[IsProjectManager])
    def release(self, request, pk=None):
        """Release an employee from a project assignment."""
        assignment = self.get_object()
        if assignment.status == 'Released':
            return Response({'error': 'Already released.'}, status=status.HTTP_400_BAD_REQUEST)
        assignment.status = 'Released'
        assignment.released_at = timezone.now()
        assignment.save(update_fields=['status', 'released_at', 'updated_at'])
        return Response(ProjectAssignmentSerializer(assignment).data)

    @action(detail=False, methods=['get'])
    def available_employees(self, request):
        """
        Returns employees NOT currently on an active/planned assignment.
        Optional filter: trade_id, skill_level.
        """
        trade_id = request.query_params.get('trade_id')
        skill_level = request.query_params.get('skill_level')

        busy_ids = ProjectAssignment.objects.filter(
            status__in=['Planned', 'Active']
        ).values_list('employee_id', flat=True)

        employees = Employee.objects.exclude(id__in=busy_ids).select_related('hr_profile__trade')

        if trade_id:
            employees = employees.filter(hr_profile__trade_id=trade_id)
        if skill_level:
            employees = employees.filter(hr_profile__skill_level=skill_level)

        serializer = EmployeeWithHRProfileSerializer(employees, many=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# ATTENDANCE EXTENSION (Draft / Approve / Reject)
# ---------------------------------------------------------------------------

class AttendanceExtensionViewSet(viewsets.ModelViewSet):
    queryset = AttendanceExtension.objects.select_related(
        'attendance__user', 'project', 'activity', 'approved_by'
    ).all()
    serializer_class = AttendanceExtensionSerializer

    def get_permissions(self):
        if self.action in ('approve', 'reject', 'bulk_approve'):
            return [IsProjectManager()]
        if self.action in ('create', 'bulk_create'):
            return [IsForeman()]
        return [IsHROfficer()]

    def get_queryset(self):
        qs = AttendanceExtension.objects.select_related(
            'attendance__user', 'project', 'activity', 'approved_by'
        ).all()
        project_id = self.request.query_params.get('project_id')
        approval_status = self.request.query_params.get('approval_status')
        date = self.request.query_params.get('date')
        employee_id = self.request.query_params.get('employee_id')

        if project_id:
            qs = qs.filter(project_id=project_id)
        if approval_status:
            qs = qs.filter(approval_status=approval_status)
        if date:
            qs = qs.filter(attendance__date=date)
        if employee_id:
            qs = qs.filter(attendance__user_id=employee_id)
        return qs

    @action(detail=True, methods=['post'], permission_classes=[IsProjectManager])
    def approve(self, request, pk=None):
        """Approve a single attendance extension → triggers cost posting via signal."""
        ext = self.get_object()
        if ext.approval_status == 'Approved':
            return Response({'error': 'Already approved.'}, status=status.HTTP_400_BAD_REQUEST)
        ext.approval_status = 'Approved'
        ext.approved_by = request.user
        ext.approved_at = timezone.now()
        ext.rejection_reason = None
        ext.save(update_fields=['approval_status', 'approved_by', 'approved_at', 'rejection_reason', 'updated_at'])
        return Response(AttendanceExtensionSerializer(ext).data)

    @action(detail=True, methods=['post'], permission_classes=[IsProjectManager])
    def reject(self, request, pk=None):
        """Reject attendance extension."""
        ext = self.get_object()
        reason = request.data.get('reason', '')
        ext.approval_status = 'Rejected'
        ext.rejection_reason = reason
        ext.approved_by = request.user
        ext.approved_at = timezone.now()
        ext.save(update_fields=['approval_status', 'rejection_reason', 'approved_by', 'approved_at', 'updated_at'])
        return Response(AttendanceExtensionSerializer(ext).data)

    @action(detail=False, methods=['post'], permission_classes=[IsProjectManager])
    def bulk_approve(self, request):
        """Bulk approve attendance extensions. Body: { "ids": [1, 2, 3] }"""
        ids = request.data.get('ids', [])
        if not ids:
            return Response({'error': 'ids list is required'}, status=status.HTTP_400_BAD_REQUEST)

        now = timezone.now()
        updated = AttendanceExtension.objects.filter(
            id__in=ids, approval_status='Draft'
        ).update(
            approval_status='Approved',
            approved_by=request.user,
            approved_at=now,
        )
        # Signals won't fire on bulk update — manually trigger cost posting
        from .signals import post_labor_cost
        for ext in AttendanceExtension.objects.filter(id__in=ids, approval_status='Approved', cost_posted=False):
            post_labor_cost(ext)

        return Response({'approved_count': updated})

    @action(detail=False, methods=['post'], permission_classes=[IsForeman])
    def bulk_create(self, request):
        """
        Foreman daily grid entry.
        Creates AttendanceExtension records for multiple employees in one call.
        """
        serializer = AttendanceExtensionBulkCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        project = data['project']
        activity = data.get('activity')
        date = data['date']
        source = data['source']
        entries = data['entries']

        created_ids = []
        errors = []

        with transaction.atomic():
            for entry in entries:
                att_id = entry.get('attendance_id')
                total_hours = entry.get('total_hours', 0)
                try:
                    attendance = Attendance.objects.get(id=att_id, date=date)
                except Attendance.DoesNotExist:
                    errors.append({'attendance_id': att_id, 'error': 'Attendance record not found for this date.'})
                    continue

                ext, created = AttendanceExtension.objects.update_or_create(
                    attendance=attendance,
                    defaults={
                        'project': project,
                        'activity': activity,
                        'total_hours': total_hours,
                        'source': source,
                        'approval_status': 'Draft',
                    }
                )
                created_ids.append(ext.id)

        return Response({
            'created_or_updated': len(created_ids),
            'ids': created_ids,
            'errors': errors,
        }, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# ACTUAL COST (read-only for Finance; auto-posted by signals)
# ---------------------------------------------------------------------------

class ActualCostViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ActualCost.objects.select_related('project', 'activity', 'employee').all()
    serializer_class = ActualCostSerializer
    permission_classes = [IsFinance]

    def get_queryset(self):
        qs = ActualCost.objects.select_related('project', 'activity', 'employee').all()
        project_id = self.request.query_params.get('project_id')
        employee_id = self.request.query_params.get('employee_id')
        date_from = self.request.query_params.get('date_from')
        date_to = self.request.query_params.get('date_to')
        cbs_code = self.request.query_params.get('cbs_code')

        if project_id:
            qs = qs.filter(project_id=project_id)
        if employee_id:
            qs = qs.filter(employee_id=employee_id)
        if date_from:
            qs = qs.filter(cost_date__gte=date_from)
        if date_to:
            qs = qs.filter(cost_date__lte=date_to)
        if cbs_code:
            qs = qs.filter(cbs_code=cbs_code)
        return qs

    @action(detail=False, methods=['get'])
    def summary_by_project(self, request):
        """Finance: total labor cost per project."""
        data = (
            ActualCost.objects.values('project__project_code', 'project__project_name')
            .annotate(
                total_cost=Sum('total_cost'),
                total_hours=Sum('regular_hours'),
                total_ot_hours=Sum('overtime_hours'),
                records=Count('id'),
            )
            .order_by('-total_cost')
        )
        return Response(list(data))

    @action(detail=False, methods=['get'])
    def summary_by_trade(self, request):
        """Finance: total labor cost per trade."""
        project_id = request.query_params.get('project_id')
        qs = ActualCost.objects.all()
        if project_id:
            qs = qs.filter(project_id=project_id)
        data = (
            qs.values('employee__hr_profile__trade__trade_name')
            .annotate(
                total_cost=Sum('total_cost'),
                total_hours=Sum('regular_hours'),
            )
            .order_by('-total_cost')
        )
        return Response(list(data))


# ---------------------------------------------------------------------------
# PRODUCTIVITY
# ---------------------------------------------------------------------------

class ProductivityRecordViewSet(viewsets.ModelViewSet):
    queryset = ProductivityRecord.objects.select_related('project', 'activity', 'trade').all()
    serializer_class = ProductivityRecordSerializer
    permission_classes = [IsHROfficer]

    def perform_create(self, serializer):
        serializer.save(entered_by=self.request.user)

    def get_queryset(self):
        qs = ProductivityRecord.objects.select_related('project', 'activity', 'trade').all()
        project_id = self.request.query_params.get('project_id')
        activity_id = self.request.query_params.get('activity_id')
        trade_id = self.request.query_params.get('trade_id')
        date_from = self.request.query_params.get('date_from')
        date_to = self.request.query_params.get('date_to')
        if project_id:
            qs = qs.filter(project_id=project_id)
        if activity_id:
            qs = qs.filter(activity_id=activity_id)
        if trade_id:
            qs = qs.filter(trade_id=trade_id)
        if date_from:
            qs = qs.filter(record_date__gte=date_from)
        if date_to:
            qs = qs.filter(record_date__lte=date_to)
        return qs

    @action(detail=False, methods=['get'])
    def kpis(self, request):
        """Productivity KPIs for a project."""
        project_id = request.query_params.get('project_id')
        if not project_id:
            return Response({'error': 'project_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        data = (
            ProductivityRecord.objects.filter(project_id=project_id)
            .values('activity__activity_name', 'trade__trade_name')
            .annotate(
                total_man_hours=Sum('man_hours_used'),
                total_qty=Sum('quantity_executed'),
                avg_productivity=Sum('quantity_executed') / Sum('man_hours_used'),
            )
            .order_by('-avg_productivity')
        )
        return Response(list(data))


# ---------------------------------------------------------------------------
# CAMP MANAGEMENT
# ---------------------------------------------------------------------------

class CampViewSet(viewsets.ModelViewSet):
    queryset = Camp.objects.all()
    permission_classes = [IsHRAdminOrReadOnly]

    def get_serializer_class(self):
        if self.action == 'list':
            return CampListSerializer
        return CampSerializer

    @action(detail=False, methods=['get'])
    def occupancy_dashboard(self, request):
        """Heatmap data: occupancy status per camp."""
        camps = Camp.objects.filter(is_active=True)
        result = []
        for camp in camps:
            occupied = camp.occupied_beds
            result.append({
                'camp_id': camp.id,
                'camp_name': camp.name,
                'location': camp.location,
                'total_capacity': camp.capacity,
                'occupied': occupied,
                'available': camp.capacity - occupied,
                'occupancy_percent': round((occupied / camp.capacity * 100), 1) if camp.capacity else 0,
            })
        serializer = CampOccupancySerializer(result, many=True)
        return Response(serializer.data)


class CampRoomViewSet(viewsets.ModelViewSet):
    queryset = CampRoom.objects.select_related('camp').all()
    serializer_class = CampRoomSerializer
    permission_classes = [IsHRAdminOrReadOnly]

    def get_queryset(self):
        qs = CampRoom.objects.select_related('camp').all()
        camp_id = self.request.query_params.get('camp_id')
        if camp_id:
            qs = qs.filter(camp_id=camp_id)
        return qs


class CampAllocationViewSet(viewsets.ModelViewSet):
    queryset = CampAllocation.objects.select_related('employee', 'room__camp').all()
    serializer_class = CampAllocationSerializer
    permission_classes = [IsHROfficer]

    def get_queryset(self):
        qs = CampAllocation.objects.select_related('employee', 'room__camp').all()
        camp_id = self.request.query_params.get('camp_id')
        employee_id = self.request.query_params.get('employee_id')
        alloc_status = self.request.query_params.get('status')
        if camp_id:
            qs = qs.filter(room__camp_id=camp_id)
        if employee_id:
            qs = qs.filter(employee_id=employee_id)
        if alloc_status:
            qs = qs.filter(status=alloc_status)
        return qs

    @action(detail=True, methods=['post'], permission_classes=[IsHROfficer])
    def release(self, request, pk=None):
        """Release a camp bed allocation."""
        allocation = self.get_object()
        if allocation.status == 'Released':
            return Response({'error': 'Already released.'}, status=status.HTTP_400_BAD_REQUEST)
        allocation.status = 'Released'
        allocation.released_at = timezone.now()
        allocation.end_date = timezone.now().date()
        allocation.save(update_fields=['status', 'released_at', 'end_date'])
        return Response(CampAllocationSerializer(allocation).data)


# ---------------------------------------------------------------------------
# ONBOARDING WIZARD (HR Admin workflow)
# ---------------------------------------------------------------------------

class OnboardingView(APIView):
    """
    Step-by-step onboarding endpoint.
    POST with step: 'create_profile' | 'upload_docs' | 'assign_trade' | 'allocate_camp' | 'assign_project'
    """
    permission_classes = [IsHRAdmin]

    def post(self, request):
        step = request.data.get('step')

        if step == 'create_profile':
            return self._create_profile(request)
        elif step == 'assign_trade':
            return self._assign_trade(request)
        elif step == 'validate_documents':
            return self._validate_documents(request)
        elif step == 'allocate_camp':
            return self._allocate_camp(request)
        elif step == 'assign_project':
            return self._assign_project(request)
        else:
            return Response(
                {'error': 'Invalid step. Use: create_profile, assign_trade, validate_documents, allocate_camp, assign_project'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def _create_profile(self, request):
        """Create or update EmployeeHRProfile for an existing Employee."""
        employee_id = request.data.get('employee_id')
        try:
            employee = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=status.HTTP_404_NOT_FOUND)

        serializer = EmployeeHRProfileSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        profile, _ = EmployeeHRProfile.objects.update_or_create(
            employee=employee,
            defaults={
                'employee_code': serializer.validated_data.get('employee_code'),
                'employment_type': serializer.validated_data.get('employment_type', 'Direct'),
                'basic_rate': serializer.validated_data.get('basic_rate'),
                'overtime_rate_multiplier': serializer.validated_data.get('overtime_rate_multiplier', 1.5),
                'cost_center_default': serializer.validated_data.get('cost_center_default'),
            }
        )
        return Response(EmployeeHRProfileSerializer(profile).data, status=status.HTTP_201_CREATED)

    def _assign_trade(self, request):
        employee_id = request.data.get('employee_id')
        trade_id = request.data.get('trade_id')
        skill_level = request.data.get('skill_level')
        try:
            profile = EmployeeHRProfile.objects.get(employee_id=employee_id)
        except EmployeeHRProfile.DoesNotExist:
            return Response({'error': 'HR Profile not found. Complete create_profile step first.'}, status=status.HTTP_404_NOT_FOUND)

        profile.trade_id = trade_id
        profile.skill_level = skill_level
        profile.save(update_fields=['trade_id', 'skill_level', 'updated_at'])
        return Response(EmployeeHRProfileSerializer(profile).data)

    def _validate_documents(self, request):
        employee_id = request.data.get('employee_id')
        dms_ids = request.data.get('dms_document_ids', [])
        try:
            profile = EmployeeHRProfile.objects.get(employee_id=employee_id)
        except EmployeeHRProfile.DoesNotExist:
            return Response({'error': 'HR Profile not found.'}, status=status.HTTP_404_NOT_FOUND)

        profile.documents_dms_ids = dms_ids
        profile.documents_valid = bool(dms_ids)
        profile.save(update_fields=['documents_dms_ids', 'documents_valid', 'updated_at'])
        return Response({'documents_valid': profile.documents_valid, 'dms_ids': profile.documents_dms_ids})

    def _allocate_camp(self, request):
        employee_id = request.data.get('employee_id')
        room_id = request.data.get('room_id')
        bed_number = request.data.get('bed_number')
        start_date = request.data.get('start_date')

        serializer = CampAllocationSerializer(data={
            'employee': employee_id,
            'room': room_id,
            'bed_number': bed_number,
            'start_date': start_date,
        })
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        allocation = serializer.save()
        return Response(CampAllocationSerializer(allocation).data, status=status.HTTP_201_CREATED)

    def _assign_project(self, request):
        employee_id = request.data.get('employee_id')
        project_id = request.data.get('project_id')
        activity_id = request.data.get('activity_id')
        start_date = request.data.get('start_date')
        end_date = request.data.get('end_date')

        serializer = ProjectAssignmentSerializer(data={
            'employee': employee_id,
            'project': project_id,
            'activity': activity_id,
            'start_date': start_date,
            'end_date': end_date,
        }, context={'request': request})

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        assignment = serializer.save(assigned_by=request.user)
        return Response(ProjectAssignmentSerializer(assignment).data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# OFFBOARDING (HR Admin workflow)
# ---------------------------------------------------------------------------

class OffboardingView(APIView):
    """
    Offboard an employee:
    1. Release all project assignments
    2. Release camp bed
    3. Archive employee status
    """
    permission_classes = [IsHRAdmin]

    def post(self, request):
        employee_id = request.data.get('employee_id')
        if not employee_id:
            return Response({'error': 'employee_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            employee = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=status.HTTP_404_NOT_FOUND)

        with transaction.atomic():
            now = timezone.now()

            # 1. Release project assignments
            released_assignments = ProjectAssignment.objects.filter(
                employee=employee, status__in=['Planned', 'Active']
            ).update(status='Released', released_at=now)

            # 2. Release camp allocations
            released_camps = CampAllocation.objects.filter(
                employee=employee, status='Active'
            ).update(status='Released', released_at=now, end_date=now.date())

            # 3. Archive employee (update existing record's status field — read-only action on existing model)
            employee.status = 'Offboarded'
            employee.save(update_fields=['status'])

        return Response({
            'employee': employee.name,
            'assignments_released': released_assignments,
            'camp_beds_released': released_camps,
            'status': 'Offboarded',
        })


# ---------------------------------------------------------------------------
# TRANSFER BETWEEN PROJECTS (HR Admin workflow)
# ---------------------------------------------------------------------------

class ProjectTransferView(APIView):
    """
    Transfer employee from one project to another:
    1. Release old assignment
    2. Close open attendance extensions as Draft warning
    3. Create new assignment
    4. Optionally reassign camp
    """
    permission_classes = [IsProjectManager]

    def post(self, request):
        employee_id = request.data.get('employee_id')
        new_project_id = request.data.get('new_project_id')
        new_activity_id = request.data.get('new_activity_id')
        new_start_date = request.data.get('new_start_date')
        new_end_date = request.data.get('new_end_date')

        if not all([employee_id, new_project_id, new_start_date]):
            return Response(
                {'error': 'employee_id, new_project_id, new_start_date are required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            employee = Employee.objects.get(id=employee_id)
        except Employee.DoesNotExist:
            return Response({'error': 'Employee not found'}, status=status.HTTP_404_NOT_FOUND)

        with transaction.atomic():
            now = timezone.now()

            # Release current active assignment
            old_released = ProjectAssignment.objects.filter(
                employee=employee, status__in=['Planned', 'Active']
            ).update(status='Released', released_at=now)

            # Create new assignment
            serializer = ProjectAssignmentSerializer(data={
                'employee': employee_id,
                'project': new_project_id,
                'activity': new_activity_id,
                'start_date': new_start_date,
                'end_date': new_end_date,
            }, context={'request': request})
            serializer.is_valid(raise_exception=True)
            new_assignment = serializer.save(assigned_by=request.user)

        return Response({
            'employee': employee.name,
            'old_assignments_released': old_released,
            'new_assignment': ProjectAssignmentSerializer(new_assignment).data,
        }, status=status.HTTP_201_CREATED)
