# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-02-27 16:24
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0002_auto_20170227_1014'),
    ]

    operations = [
        migrations.RenameField(
            model_name='training_model',
            old_name='trainging_model_name',
            new_name='training_model_name',
        ),
    ]
