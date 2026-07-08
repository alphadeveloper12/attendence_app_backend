from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import FaceTemplate, Employee, Site, AdminProfile
from .engine import ENGINE
import numpy as np


@receiver(post_save, sender=Site)
def add_new_site_to_viewers(sender, instance, created, **kwargs):
    """Read-only Viewers are scoped to ALL sites (that's how they see everything).
    When a new site is created, attach it to every viewer so their view stays
    complete without any per-endpoint scoping changes."""
    if not created:
        return
    try:
        for prof in AdminProfile.objects.filter(role=AdminProfile.ROLE_VIEWER):
            prof.sites.add(instance)
    except Exception:
        pass

@receiver(post_save, sender=FaceTemplate)
def update_index_on_save(sender, instance, **kwargs):
    emb = np.array(instance.embedding, dtype=np.float32)
    ENGINE.update_or_add(instance.id, instance.employee_id, instance.employee.site_id, emb)

@receiver(post_delete, sender=FaceTemplate)
def update_index_on_delete(sender, instance, **kwargs):
    ENGINE.remove(instance.id)

# When True, the Employee post_save index sync below is skipped. Bulk operations
# set this so every emp.save() doesn't rebuild the ENTIRE FAISS index (which is
# O(all templates) per call → catastrophic for large imports). They rebuild the
# index once at the end instead.
suspend_employee_index_sync = False


@receiver(post_save, sender=Employee)
def update_index_on_employee_change(sender, instance, **kwargs):
    # If site changes, we need to update all templates for this employee
    # This is a bit heavy but necessary if we partition by site.
    # Ideally we check if site actually changed.
    # For now, just re-add all templates for this employee.
    if suspend_employee_index_sync:
        return
    for t in instance.templates.all():
        emb = np.array(t.embedding, dtype=np.float32)
        ENGINE.update_or_add(t.id, instance.id, instance.site_id, emb)