from rest_framework import serializers
from base.models import Solved
        
class SolvedSerializer(serializers.ModelSerializer):
    class Meta:
        model = Solved
        fields = '__all__'