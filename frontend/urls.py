from django.urls import path,include
from .views import index,ratel,ind,eva

urlpatterns = [
    path('',index),
    path('Rating_List/',ratel),
    path('Index/',ind),
    path('Evaluation/',eva),
]

