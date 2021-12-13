from rest_framework import serializers
from .models import Solved, TestBlob
        
class SolvedSerializer(serializers.ModelSerializer):
    class Meta:
        model = Solved
        fields = '__all__'
        
class BlobSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestBlob
        fields = '__all__'