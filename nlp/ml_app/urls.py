from django.urls import path
from . import views

urlpatterns = [
    path('',views.predict),
    path('details',views.getDetails),
    path('predict',views.predict)
]