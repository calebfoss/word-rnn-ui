from django.db import models

class Training_Input(models.Model):
    training_input_name = models.CharField(max_length=50)
    directory = models.CharField(max_length=100)
    def __str__(self):
        return self.training_input_name