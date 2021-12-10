from django.forms import ModelForm
from .models import Solved
        
class SolvedForm(ModelForm):
    class Meta:
        model = Solved
        fields = '__all__'