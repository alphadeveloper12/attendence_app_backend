"""Move Emirates ID values that were typed into the Email field into `emirates_id`.

Before the dedicated Emirates ID field existed, operators put the EID in the
Email box. This walks every employee, and where the email looks like a UAE
Emirates ID (784-YYYY-NNNNNNN-C, with or without dashes) copies it across.

Dry run by default — nothing is written unless you pass --apply:

    python manage.py migrate_emirates_id                 # preview
    python manage.py migrate_emirates_id --apply         # write emirates_id
    python manage.py migrate_emirates_id --apply --clear-email   # ...and blank the email
"""
import re

from django.core.management.base import BaseCommand

from attendance import signals as _signals
from attendance.models import Employee

EID_RE = re.compile(r"^784-?\d{4}-?\d{7}-?\d$")


class Command(BaseCommand):
    help = "Copy Emirates-ID-looking values out of Employee.email into Employee.emirates_id."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true",
                            help="Actually write the changes (default: dry run).")
        parser.add_argument("--clear-email", action="store_true",
                            help="Also blank the email field after moving the value.")

    def handle(self, *args, **options):
        do_write = options["apply"]
        clear_email = options["clear_email"]

        # Changing employees fires the FAISS re-index signal per save; none of
        # these edits affect the face index, so suspend it for the whole run.
        _signals.suspend_employee_index_sync = True
        moved = already = 0
        try:
            qs = Employee.objects.exclude(email__isnull=True).exclude(email="")
            for emp in qs.iterator():
                raw = (emp.email or "").strip()
                if not EID_RE.match(raw):
                    continue
                if emp.emirates_id:
                    already += 1
                    continue
                moved += 1
                if do_write:
                    emp.emirates_id = raw
                    fields = ["emirates_id"]
                    if clear_email:
                        emp.email = None
                        fields.append("email")
                    emp.save(update_fields=fields)
                else:
                    self.stdout.write(
                        f"[dry-run] {emp.badge_number or emp.id} — {emp.name}: '{raw}' → emirates_id"
                    )
        finally:
            _signals.suspend_employee_index_sync = False

        verb = "Moved" if do_write else "Would move"
        self.stdout.write(self.style.SUCCESS(
            f"{verb} {moved} Emirates ID(s). Skipped {already} that already had one."
        ))
        if not do_write:
            self.stdout.write("Re-run with --apply to write the changes.")
