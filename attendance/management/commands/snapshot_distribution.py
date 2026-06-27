"""Freeze the current Distribution List so past dates stay viewable.

Run once a day (e.g. just before midnight) via cron:

    0 23 * * *  cd /path/to/project && venv/bin/python manage.py snapshot_distribution

Each run stores the all-sites distribution for the three tab types
(resource / staff / worker) for the given date (default: today).
"""
from datetime import datetime

from django.core.management.base import BaseCommand
from django.utils import timezone

from attendance.views import capture_distribution_snapshots


class Command(BaseCommand):
    help = "Store a snapshot of the manpower distribution for a date (default today)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--date', type=str, default=None,
            help="Date to stamp the snapshot with, YYYY-MM-DD (default: today).",
        )

    def handle(self, *args, **options):
        for_date = timezone.localdate()
        if options.get('date'):
            try:
                for_date = datetime.strptime(options['date'], '%Y-%m-%d').date()
            except ValueError:
                self.stderr.write(self.style.ERROR("Invalid --date, use YYYY-MM-DD."))
                return
        saved = capture_distribution_snapshots(for_date)
        self.stdout.write(self.style.SUCCESS(
            f"Saved {saved}/3 distribution snapshots for {for_date}."
        ))
