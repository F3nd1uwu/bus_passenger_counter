from datetime import timedelta

from django import forms
from django.contrib.auth import password_validation
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import PermissionDenied
from django.utils import timezone

from .models import Bus, User


class PeriodForm(forms.Form):
    date_from = forms.DateField(label='С', widget=forms.DateInput(attrs={'type': 'date'}))
    date_to = forms.DateField(label='По', widget=forms.DateInput(attrs={'type': 'date'}))
    bus = forms.CharField(label='Автобус', required=False, max_length=32)
    mode = forms.ChoiceField(label='Данные', choices=[('all', 'Все данные'), ('production', 'Рабочие'),
                                                    ('test', 'Тестовые'), ('unknown', 'Не проверены')])

    def clean(self):
        data = super().clean()
        first, last = data.get('date_from'), data.get('date_to')
        if first and last and (first > last or last - first > timedelta(days=365)):
            raise forms.ValidationError('Выберите период до 366 дней, начало не позже окончания.')
        return data


def period_form(request):
    today = timezone.localdate()
    defaults = {'date_from': (today - timedelta(days=6)).isoformat(), 'date_to': today.isoformat(),
                'bus': '', 'mode': 'all'}
    supplied = request.POST if request.method == 'POST' else request.GET
    form = PeriodForm({**defaults, **{key: supplied[key] for key in defaults if key in supplied}})
    if form.is_valid() and not request.user.is_console_admin:
        selected = form.cleaned_data['bus']
        if selected and not request.user.buses.filter(bus_id=selected).exists():
            raise PermissionDenied
    return form


class ConsoleUserCreateForm(UserCreationForm):
    buses = forms.ModelMultipleChoiceField(label='Доступные автобусы', queryset=Bus.objects.all(), required=False,
                                           widget=forms.CheckboxSelectMultiple)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'role', 'buses']
        labels = {'role': 'Роль'}

    def save(self, commit=True):
        user = super().save(commit=commit)
        if commit:
            user.buses.set(self.cleaned_data['buses'])
        return user


class ConsoleUserEditForm(forms.ModelForm):
    buses = forms.ModelMultipleChoiceField(label='Доступные автобусы', queryset=Bus.objects.all(), required=False,
                                           widget=forms.CheckboxSelectMultiple)
    password1 = forms.CharField(label='Новый пароль', required=False, strip=False, widget=forms.PasswordInput)
    password2 = forms.CharField(label='Повторите новый пароль', required=False, strip=False, widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'role', 'is_active', 'buses']
        labels = {'role': 'Роль'}

    def __init__(self, *args, actor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.fields['buses'].initial = self.instance.buses.all()
        if self.instance.is_superuser:
            self.fields['role'].disabled = True

    def clean(self):
        data = super().clean()
        if self.actor and self.actor.pk == self.instance.pk:
            if not data.get('is_active') or (data.get('role') != User.Role.ADMIN and not self.instance.is_superuser):
                raise forms.ValidationError('Нельзя отключить собственный административный доступ.')
        password = data.get('password1')
        if password != data.get('password2'):
            self.add_error('password2', 'Пароли не совпадают.')
        elif password:
            try:
                password_validation.validate_password(password, self.instance)
            except forms.ValidationError as error:
                self.add_error('password1', error)
        return data

    def save(self, commit=True):
        user = super().save(commit=False)
        if self.cleaned_data.get('password1'):
            user.set_password(self.cleaned_data['password1'])
        if commit:
            user.save()
            user.buses.set(self.cleaned_data['buses'])
        return user


class BusForm(forms.ModelForm):
    class Meta:
        model = Bus
        fields = ['bus_id', 'name', 'mode', 'external_code']
        labels = {'bus_id': 'Идентификатор в программе', 'name': 'Название автобуса',
                  'mode': 'Назначение данных', 'external_code': 'Код для внешней системы (необязательно)'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields['bus_id'].disabled = True
