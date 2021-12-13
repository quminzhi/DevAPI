from django.db import models
from django.db.models.fields import CharField
import uuid
# Create your models here.
    
class Solved(models.Model):
    uid = CharField(max_length=200, null=True, blank=True)
    stylized = models.ImageField(null=True, blank=True)
    
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True,
                          primary_key=True, editable=False)
    
    def __str__(self):
        return str(self.uid)
    
class TestBlob(models.Model):
    uid = CharField(max_length=200, null=True, blank=True)
    stylized = models.ImageField(null=True, blank=True)
    load_time = models.CharField(max_length=200, null=True, blank=True)
    render_time = models.CharField(max_length=200, null=True, blank=True)
    
    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True,
                          primary_key=True, editable=False)
    
    def __str__(self):
        return str(self.uid)