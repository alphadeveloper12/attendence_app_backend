"""One-shot loader for per-site day/night shift timings.

Run on the server AFTER migrating (0048 adds the night fields):

    python manage.py seed_site_timings            # dry-run: shows what WOULD change
    python manage.py seed_site_timings --commit    # actually save

IMPORTANT — VERIFY THE VALUES in SITE_TIMINGS below before running with
--commit. They were transcribed from a printed sheet; times are 24-hour
(HH:MM). Blank night_start/night_end means the site runs a DAY shift only.

Notes on the source sheet:
  * A few sites were listed twice (a weekday row + a "Friday only" row, or a
    Staff row + a Worker row). Since a site has ONE timing record and staff &
    workers share the same window, only the primary (weekday, staff+workers)
    row is used here. Friday-only / "two shifts same day" variations can't be
    modelled per-site and are handled by each worker's Working Shift field.
  * Rows marked  # VERIFY  were hard to read on the sheet — double-check them.
"""
from django.core.management.base import BaseCommand
from attendance.models import Site


# site name : (day_start, day_end, night_start, night_end, day_off)
# night_start/night_end = None  → day-only site.
SITE_TIMINGS = {
    "Venera":         ("05:30", "17:00", None,    None,    "Friday"),
    "The Residence":  ("06:00", "16:00", "19:00", "06:00", "Sunday"),
    "Lillia":         ("06:00", "17:00", None,    None,    "Sunday"),
    "Elora":          ("05:30", "17:00", None,    None,    "Friday"),
    "Velora":         ("05:30", "17:00", None,    None,    "Friday"),
    "Alana":          ("06:00", "17:00", None,    None,    "Friday"),
    "Vida":           ("05:30", "17:00", None,    None,    "Friday"),
    "Shafar Villa":   ("05:30", "17:00", "17:00", "04:30", "Sunday"),
    "Club House":     ("05:30", "17:00", None,    None,    "Friday"),
    "Opal Garden":    ("06:00", "16:00", None,    None,    "Sunday"),
    "City Walk 5.6":  ("06:00", "19:00", "16:00", "06:00", "Friday"),
    "Avenue Mall":    ("06:00", "19:00", None,    None,    "Sunday"),
    "City Walk 3.2":  ("06:00", "19:00", "19:00", "05:00", "Friday"),
    "Factory -KFD":   ("07:00", "17:00", "19:00", "05:00", "Sunday"),
    "Plant Workshop": ("05:30", "17:30", "18:00", "05:00", "Friday"),
    "Factory -KAMI":  ("07:00", "17:00", "19:00", "05:00", "Sunday"),
    "Head Office":    ("07:30", "16:45", None,    None,    "Sunday"),
    "Rivera":         ("05:30", "17:30", "15:00", "02:00", "Friday"),
    "Maryah Plaza":   ("06:00", "17:00", "18:00", "06:00", "Sunday"),  # VERIFY night window
    "Marwan Villa":   ("06:00", "17:00", None,    None,    "Sunday"),
}


class Command(BaseCommand):
    help = "Load per-site day/night shift timings (dry-run unless --commit)."

    def add_arguments(self, parser):
        parser.add_argument("--commit", action="store_true",
                            help="Actually save changes (default is a dry run).")

    def handle(self, *args, **opts):
        commit = opts["commit"]
        updated, missing = 0, []

        for name, (d_start, d_end, n_start, n_end, day_off) in SITE_TIMINGS.items():
            site = (Site.objects.filter(name__iexact=name).first()
                    or Site.objects.filter(name__icontains=name).first())
            if not site:
                missing.append(name)
                continue

            site.worker_start_time = d_start
            site.worker_end_time = d_end
            site.office_start_time = d_start   # staff & workers share the window
            site.office_end_time = d_end
            site.night_start_time = n_start
            site.night_end_time = n_end
            site.worker_day_off = day_off
            site.office_day_off = day_off

            night = f"{n_start}-{n_end}" if n_start else "day-only"
            self.stdout.write(f"  {site.name:20s} day {d_start}-{d_end}  night {night}  off {day_off}")
            if commit:
                site.save()
            updated += 1

        self.stdout.write("")
        if missing:
            self.stdout.write(self.style.WARNING(
                f"Not found (check exact site name): {', '.join(missing)}"))
        verb = "Saved" if commit else "Would update (dry run)"
        self.stdout.write(self.style.SUCCESS(f"{verb}: {updated} site(s)."))
        if not commit:
            self.stdout.write("Re-run with --commit to save.")
