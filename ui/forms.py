from django import forms

class TrainingModelForm(forms.Form):
	training_model_selection = forms.CharField(label='Training model',max_length=50)