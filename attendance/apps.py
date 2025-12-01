# attendance/apps.py
import os, sys
from django.apps import AppConfig

SKIP_CMDS = {"makemigrations", "migrate", "collectstatic", "shell", "dbshell"}

class AttendanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "attendance"

    def ready(self):
        # activate signals
        import attendance.signals

        if os.environ.get("SKIP_FACE_WARMUP") == "1":
            return
        if any(cmd in sys.argv for cmd in SKIP_CMDS):
            return

        try:
            from .engine import ENGINE
            from .models import FaceTemplate
            import numpy as np

            qs = FaceTemplate.objects.all().only("id", "employee_id", "embedding")
            tuples = [
                (t.id, t.employee_id, np.array(t.embedding, dtype=np.float32))
                for t in qs
            ]
            ENGINE.rebuild_index(tuples)
        except Exception as e:
            print("Index rebuild error:", e)
