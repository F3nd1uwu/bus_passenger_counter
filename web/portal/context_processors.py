from django.conf import settings


def console_context(request):
    return {'demo_mode': settings.DEMO, 'console_timezone': settings.TIME_ZONE}
