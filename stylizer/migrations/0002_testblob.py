# Generated by Django 3.2.9 on 2021-12-13 02:28

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('stylizer', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TestBlob',
            fields=[
                ('uid', models.CharField(blank=True, max_length=200, null=True)),
                ('stylized', models.ImageField(blank=True, null=True, upload_to='')),
                ('load_time', models.CharField(blank=True, max_length=200, null=True)),
                ('render_time', models.CharField(blank=True, max_length=200, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, unique=True)),
            ],
        ),
    ]
