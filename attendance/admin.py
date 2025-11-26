from django.contrib import admin
from .models import Employee, Attendance
from django.utils import timezone
from datetime import timedelta


# Attendance Admin
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'status', 'check_in_time', 'check_out_time', 'late_minutes', 'early_minutes', 'latitude', 'longitude')
    list_filter = ('date', 'status', 'user__site')  # Filter by date, status, and site
    search_fields = ['user__name', 'user__email', 'user__phone']
    list_per_page = 20

    def get_queryset(self, request):
        queryset = super().get_queryset(request)

        # Adding the ability to filter by weekly or monthly attendance
        today = timezone.now().date()
        if 'weekly' in request.GET:
            start_of_week = today - timedelta(days=today.weekday())  # Start of the current week
            end_of_week = start_of_week + timedelta(days=6)  # End of the current week
            queryset = queryset.filter(date__range=[start_of_week, end_of_week])
        elif 'monthly' in request.GET:
            start_of_month = today.replace(day=1)  # Start of the current month
            queryset = queryset.filter(date__month=today.month, date__year=today.year)

        return queryset

    # To get total late minutes for a user
    def late_minutes(self, obj):
        return obj.late_minutes

    # To get total early minutes for a user
    def early_minutes(self, obj):
        return obj.early_minutes

    late_minutes.admin_order_field = 'late_minutes'
    early_minutes.admin_order_field = 'early_minutes'


# Employee Admin
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'phone', 'department', 'position', 'site', 'get_late_minutes')
    list_filter = ('site', 'department', 'position')  # Admin can filter by site, department, and position
    search_fields = ['name', 'email', 'phone']
    ordering = ('site',)  # Default sorting based on the site field
    list_per_page = 20

    def get_late_minutes(self, obj):
        # Total late minutes for each user
        total_late_minutes = Attendance.objects.filter(user=obj).aggregate(Sum('late_minutes'))['late_minutes__sum']
        return total_late_minutes if total_late_minutes else 0

    get_late_minutes.admin_order_field = 'late_minutes'
    get_late_minutes.short_description = 'Total Late Minutes'


# Register models in admin
admin.site.register(Employee, EmployeeAdmin)
admin.site.register(Attendance, AttendanceAdmin)
