# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-27 16:40
from __future__ import unicode_literals

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0003_auto_20170227_1024'),
    ]

    operations = [
        migrations.AddField(
            model_name='training_model',
            name='pub_date',
            field=models.DateTimeField(default=django.utils.timezone.now, verbose_name='date published'),
            preserve_default=False,
        ),
    ]
