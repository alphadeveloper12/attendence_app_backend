from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import FaceTemplate
from .engine import ENGINE
import numpy as np

@receiver(post_save, sender=FaceTemplate)
def update_index_on_save(sender, instance, **kwargs):
    emb = np.array(instance.embedding, dtype=np.float32)
    ENGINE.update_or_add(instance.id, instance.employee_id, emb)

@receiver(post_delete, sender=FaceTemplate)
def update_index_on_delete(sender, instance, **kwargs):
    ENGINE.remove(instance.id)