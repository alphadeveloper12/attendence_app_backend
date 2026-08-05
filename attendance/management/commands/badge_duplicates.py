"""Find and fix employees sharing the same Badge ID.

Badge ID is the unique identifier across the system (imports key on it and the
API now rejects duplicates), but historical data may still contain clashes.

Usage on the server:

    # 1. See every duplicate group (safe, read-only)
    python manage.py badge_duplicates

    # 2a. Keep both people but blank one badge (safe, no data loss)
    python manage.py badge_duplicates --clear 1234

    # 2b. Same physical person entered twice — merge the duplicate into the
    #     keeper and delete it. Attendance, history, attachments, salary
    #     history, face templates and notes are moved across first.
    python manage.py badge_duplicates --merge KEEP_ID REMOVE_ID
"""
from collections import defaultdict

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from attendance.models import Attendance, Employee


def _norm_badge(b):
    if not b:
        return ''
    s = str(b).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s.replace(' ', '').lower()


class Command(BaseCommand):
    help = "List or fix employees that share the same Badge ID."

    def add_arguments(self, parser):
        parser.add_argument('--clear', type=int, metavar='EMPLOYEE_ID',
                            help="Blank this employee's badge number (keeps the employee).")
        parser.add_argument('--merge', nargs=2, type=int, metavar=('KEEP_ID', 'REMOVE_ID'),
                            help='Move all records from REMOVE_ID onto KEEP_ID, then delete REMOVE_ID.')
        parser.add_argument('--yes', action='store_true',
                            help='Skip the confirmation prompt (for scripted runs).')

    def handle(self, *args, **opts):
        if opts['clear']:
            return self._clear(opts['clear'], opts['yes'])
        if opts['merge']:
            return self._merge(opts['merge'][0], opts['merge'][1], opts['yes'])
        return self._report()

    # ── Read-only report ─────────────────────────────────────────────────
    def _report(self):
        groups = defaultdict(list)
        for e in Employee.objects.exclude(badge_number__isnull=True).exclude(badge_number='').select_related('site'):
            groups[_norm_badge(e.badge_number)].append(e)

        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        if not dupes:
            self.stdout.write(self.style.SUCCESS('No duplicate badge numbers found. Safe to add the DB unique constraint.'))
            return

        self.stdout.write(self.style.WARNING(f'{len(dupes)} badge number(s) are duplicated:\n'))
        for key, emps in sorted(dupes.items()):
            self.stdout.write(self.style.HTTP_INFO(f"Badge '{emps[0].badge_number}':"))
            for e in emps:
                att = Attendance.objects.filter(user=e)
                last = att.order_by('-date').values_list('date', flat=True).first()
                self.stdout.write(
                    f"  id={e.id:<6} {e.name:<35} site={e.site.name if e.site else '-':<20} "
                    f"status={e.status or '-':<12} attendance_rows={att.count():<5} last_attendance={last or '-'}"
                )
            self.stdout.write('')
        self.stdout.write(
            'Fix each group with ONE of:\n'
            '  python manage.py badge_duplicates --clear <id_of_wrong_record>   (keep employee, blank badge)\n'
            '  python manage.py badge_duplicates --merge <keep_id> <remove_id>  (same person twice: merge + delete)'
        )

    # ── Clear one badge ──────────────────────────────────────────────────
    def _clear(self, emp_id, yes):
        try:
            emp = Employee.objects.get(id=emp_id)
        except Employee.DoesNotExist:
            raise CommandError(f'Employee {emp_id} not found')
        if not yes:
            answer = input(f"Blank badge '{emp.badge_number}' on {emp.name} (id {emp.id})? [y/N] ")
            if answer.strip().lower() != 'y':
                self.stdout.write('Aborted.')
                return
        emp.badge_number = None
        emp.save(update_fields=['badge_number'])
        self.stdout.write(self.style.SUCCESS(f'Badge cleared on {emp.name} (id {emp.id}).'))

    # ── Merge duplicate person ───────────────────────────────────────────
    def _merge(self, keep_id, remove_id, yes):
        if keep_id == remove_id:
            raise CommandError('KEEP_ID and REMOVE_ID must differ')
        try:
            keep = Employee.objects.get(id=keep_id)
            remove = Employee.objects.get(id=remove_id)
        except Employee.DoesNotExist as exc:
            raise CommandError(str(exc))

        self.stdout.write(f'KEEP   : id={keep.id} {keep.name} (badge {keep.badge_number})')
        self.stdout.write(f'REMOVE : id={remove.id} {remove.name} (badge {remove.badge_number})')
        if not yes:
            answer = input('Move all records to KEEP and permanently delete REMOVE? [y/N] ')
            if answer.strip().lower() != 'y':
                self.stdout.write('Aborted.')
                return

        with transaction.atomic():
            # Attendance is unique per (user, date): move only dates the keeper
            # doesn't already have; the rest die with the duplicate record.
            keep_dates = set(Attendance.objects.filter(user=keep).values_list('date', flat=True))
            movable = Attendance.objects.filter(user=remove).exclude(date__in=keep_dates)
            moved = movable.update(user=keep)
            conflicts = Attendance.objects.filter(user=remove).count()

            moved_related = {}
            for rel in ('status_history', 'site_history', 'attachments',
                        'salary_history', 'templates', 'general_notes'):
                moved_related[rel] = getattr(remove, rel).update(employee=keep)

            remove.delete()

        self.stdout.write(self.style.SUCCESS(
            f'Merged. Attendance rows moved: {moved}; dropped as same-date conflicts: {conflicts}; '
            f"related records moved: {moved_related}"
        ))
        self.stdout.write(f'{keep.name} (id {keep.id}) keeps badge {keep.badge_number}.')
