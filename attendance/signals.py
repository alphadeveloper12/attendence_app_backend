from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import FaceTemplate, Employee
from .engine import ENGINE
import numpy as np

@receiver(post_save, sender=FaceTemplate)
def update_index_on_save(sender, instance, **kwargs):
    emb = np.array(instance.embedding, dtype=np.float32)
    ENGINE.update_or_add(instance.id, instance.employee_id, instance.employee.site_id, emb)

@receiver(post_delete, sender=FaceTemplate)
def update_index_on_delete(sender, instance, **kwargs):
    ENGINE.remove(instance.id)

@receiver(post_save, sender=Employee)
def update_index_on_employee_change(sender, instance, **kwargs):
    # If site changes, we need to update all templates for this employee
    # This is a bit heavy but necessary if we partition by site.
    # Ideally we check if site actually changed.
    # For now, just re-add all templates for this employee.
    for t in instance.templates.all():
        emb = np.array(t.embedding, dtype=np.float32)
        ENGINE.update_or_add(t.id, instance.id, instance.site_id, emb)