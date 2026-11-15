from datetime import timedelta
import hashlib
import hmac

from django.conf import settings
from django.contrib.auth.views import LoginView, LogoutView, PasswordChangeView
from django.db import transaction
from django.db.models import F
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache

from .models import LoginAttempt
from .services import audit


def throttle_keys(request):
    username = request.POST.get('username', '').casefold()[:150]
    ip = request.META.get('REMOTE_ADDR', '')[:100]
    return [(hmac.new(settings.SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest(), limit)
            for value, limit in [('user:' + username, 5), ('ip:' + ip, 30)]]


@method_decorator(never_cache, name='dispatch')
class ConsoleLoginView(LoginView):
    template_name = 'portal/login.html'
    redirect_authenticated_user = True

    def post(self, request, *args, **kwargs):
        cutoff = timezone.now() - timedelta(minutes=10)
        self.keys = throttle_keys(request)
        blocked = any(LoginAttempt.objects.filter(key=key, failures__gte=limit, last_failure__gte=cutoff).exists()
                      for key, limit in self.keys)
        if blocked:
            form = self.get_form_class()(request=request)
            form.cleaned_data = {}
            form.add_error(None, 'Слишком много попыток. Повторите вход через 10 минут.')
            return self.render_to_response(self.get_context_data(form=form), status=429)
        return super().post(request, *args, **kwargs)

    def form_invalid(self, form):
        now = timezone.now()
        with transaction.atomic():
            LoginAttempt.objects.filter(last_failure__lt=now - timedelta(days=1)).delete()
            for key, _ in self.keys:
                row, _ = LoginAttempt.objects.get_or_create(key=key, defaults={'last_failure': now})
                if row.last_failure < now - timedelta(minutes=10):
                    LoginAttempt.objects.filter(key=key).update(failures=0)
                LoginAttempt.objects.filter(key=key).update(failures=F('failures') + 1, last_failure=now)
        return super().form_invalid(form)

    def form_valid(self, form):
        LoginAttempt.objects.filter(key=self.keys[0][0]).delete()
        response = super().form_valid(form)
        audit(self.request.user, 'Вход в систему')
        return response


class ConsoleLogoutView(LogoutView):
    def post(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            audit(request.user, 'Выход из системы')
        return super().post(request, *args, **kwargs)


class ConsolePasswordView(PasswordChangeView):
    template_name = 'portal/form.html'
    success_url = '/'
    extra_context = {'page_title': 'Сменить пароль', 'eyebrow': 'Мой аккаунт', 'submit_label': 'Сохранить пароль'}

    def form_valid(self, form):
        response = super().form_valid(form)
        audit(self.request.user, 'Смена собственного пароля')
        return response
