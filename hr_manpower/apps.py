from django.apps import AppConfig


class HrManpowerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'hr_manpower'
    verbose_name = 'HR & Manpower Planning'

    def ready(self):
        import hr_manpower.signals  # noqa: F401
