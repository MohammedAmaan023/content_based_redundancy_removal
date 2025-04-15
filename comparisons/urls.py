from django.urls import path
from . import views

urlpatterns = [
   path('', views.home, name='home'),
    path('text/', views.process_text, name='text'),
    path('image/', views.compare_images, name='image'),
    path('compare-two/', views.compare_two_images, name='compare_two_images'),
    path('examples/', views.examples, name='example'),
    path('text/results/', views.text_results, name='text_results'),

]
