from django.urls import path
from . import views
from .auth import ConsoleLoginView, ConsoleLogoutView, ConsolePasswordView

urlpatterns = [
    path('login/', ConsoleLoginView.as_view(), name='login'),
    path('logout/', ConsoleLogoutView.as_view(), name='logout'),
    path('password/', ConsolePasswordView.as_view(), name='password'),
    path('', views.overview, name='overview'),
    path('reports/', views.reports, name='reports'),
    path('events/', views.events, name='events'),
    path('exports/', views.exports, name='exports'),
    path('exports/create/', views.create_export, name='export_create'),
    path('exports/<int:pk>/download/', views.download_export, name='export_download'),
    path('users/', views.users, name='users'),
    path('users/new/', views.user_create, name='user_create'),
    path('users/<int:pk>/', views.user_edit, name='user_edit'),
    path('buses/', views.buses, name='buses'),
    path('buses/<int:pk>/', views.bus_edit, name='bus_edit'),
    path('audit/', views.audit_log, name='audit'),
    path('health/', views.health, name='health'),
]
