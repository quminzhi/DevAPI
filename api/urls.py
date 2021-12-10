from django.urls import path
from . import views

urlpatterns = [
    path('', views.apiView, name='routes'),
    path('upload/', views.processView, name='raw'),
    path('solved/<str:uid>/', views.retrieveView, name='solved'),
]
