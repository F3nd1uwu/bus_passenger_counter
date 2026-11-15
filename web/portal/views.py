from functools import wraps
import hashlib
import logging

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import DatabaseError, transaction
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from web.analytics.models import PassengerEvent
from .forms import BusForm, ConsoleUserCreateForm, ConsoleUserEditForm, period_form
from .models import AuditEntry, Bus, ReportExport, User
from . import services

log = logging.getLogger(__name__)


def admin_required(view):
    @wraps(view)
    @login_required
    def checked(request, *args, **kwargs):
        if not request.user.is_console_admin:
            raise PermissionDenied
        return view(request, *args, **kwargs)
    return checked


def filter_context(request, title, nav):
    form = period_form(request)
    buses = Bus.objects.all() if request.user.is_console_admin else request.user.buses.all()
    return {'page_title': title, 'nav': nav, 'period_form': form, 'bus_choices': list(buses),
            'filters': form.cleaned_data if form.is_valid() else {}, 'connected': False}


def data_unavailable(context, error):
    log.warning('Passenger analytics read failed: %s', type(error).__name__)
    context['data_error'] = 'Статистика временно недоступна. Проверьте подключение к базе данных. Сохранённые данные не изменены.'


@login_required
def overview(request):
    context = filter_context(request, 'Обзор', 'overview')
    context['eyebrow'] = 'Пассажиропоток'
    if context['period_form'].is_valid():
        try:
            totals, days, recent = services.overview(request.user, context['filters'])
            context.update(totals=totals, days=days, recent=services.decorate(recent, services.registry()), connected=True)
        except DatabaseError as error:
            data_unavailable(context, error)
    return render(request, 'portal/overview.html', context, status=200 if context['period_form'].is_valid() else 400)


@login_required
def reports(request):
    context = filter_context(request, 'Отчёты', 'reports')
    context['eyebrow'] = 'Результаты обработки'
    if context['period_form'].is_valid():
        try:
            rows = services.report_rows(request.user, context['filters'])
            page = Paginator(rows, 25).get_page(request.GET.get('page'))
            page.object_list = services.decorate(page.object_list, services.registry())
            context.update(page_obj=page, connected=True)
        except DatabaseError as error:
            data_unavailable(context, error)
    return render(request, 'portal/reports.html', context, status=200 if context['period_form'].is_valid() else 400)


@login_required
def events(request):
    context = filter_context(request, 'Журнал событий', 'events')
    context['eyebrow'] = 'Проверка подсчёта'
    counted = request.GET.get('counted', 'all')
    direction = request.GET.get('direction', 'all')
    if counted not in {'all', 'yes', 'no'} or direction not in {'all', 'IN', 'OUT'}:
        return HttpResponse('Некорректный фильтр событий', status=400)
    context.update(counted_filter=counted, direction_filter=direction)
    if context['period_form'].is_valid():
        try:
            rows = services.scoped(PassengerEvent, request.user, context['filters'])
            summary = services.event_summary(request.user, context['filters'])
            if counted != 'all':
                rows = rows.filter(counted=counted == 'yes')
            if direction != 'all':
                rows = rows.filter(event_type=direction)
            rows = rows.values('id', 'bus_id', 'camera_id', 'detected_at', 'event_type', 'counted', 'reason').order_by('-detected_at', '-id')
            page = Paginator(rows, 30).get_page(request.GET.get('page'))
            page.object_list = services.decorate(page.object_list, services.registry())
            context.update(page_obj=page, event_summary=summary, connected=True)
        except DatabaseError as error:
            data_unavailable(context, error)
    return render(request, 'portal/events.html', context, status=200 if context['period_form'].is_valid() else 400)


@login_required
def exports(request):
    rows = ReportExport.objects.select_related('owner')
    if not request.user.is_console_admin:
        rows = rows.filter(owner=request.user)
    page = Paginator(rows, 20).get_page(request.GET.get('page'))
    return render(request, 'portal/exports.html', {'page_title': 'Выгрузки', 'eyebrow': 'Подготовка данных',
                                                 'nav': 'exports', 'page_obj': page})


@login_required
@require_POST
def create_export(request):
    form = period_form(request)
    if not form.is_valid():
        return HttpResponse('Некорректный период отчёта', status=400)
    try:
        source = list(services.report_rows(request.user, form.cleaned_data)[:20001])
    except DatabaseError:
        messages.error(request, 'База статистики недоступна. Отчёт не создан.')
        return redirect('reports')
    if len(source) > 20000:
        messages.error(request, 'Отчёт содержит более 20 000 строк. Уменьшите период или выберите автобус.')
        return redirect('reports')
    if not source:
        messages.error(request, 'За выбранный период нет данных для отчёта.')
        return redirect('reports')
    buses = services.registry()
    rows = [{'date': row['traffic_date'].isoformat(), 'bus_id': row['bus_id'],
             **services.bus_label(row['bus_id'], buses), 'counted_in': row['counted_in'],
             'counted_out': row['counted_out'], 'updated_at': timezone.localtime(row['last_update']).isoformat()}
            for row in source]
    digest = hashlib.sha256(services.csv_bytes(rows)).hexdigest()
    with transaction.atomic():
        report = ReportExport.objects.create(owner=request.user, date_from=form.cleaned_data['date_from'],
                  date_to=form.cleaned_data['date_to'], rows=rows, sha256=digest, row_count=len(rows))
        services.audit(request.user, 'Сформирован проверочный отчёт', report.pk)
    messages.success(request, 'Отчёт сохранён. Его можно скачать на этой странице.')
    return redirect('exports')


@login_required
def download_export(request, pk):
    report = get_object_or_404(ReportExport, pk=pk)
    if not request.user.is_console_admin:
        granted = set(request.user.buses.values_list('bus_id', flat=True))
        if report.owner_id != request.user.pk or not {row['bus_id'] for row in report.rows}.issubset(granted):
            raise PermissionDenied
    data = services.csv_bytes(report.rows)
    if hashlib.sha256(data).hexdigest() != report.sha256:
        return HttpResponse('Контрольная сумма отчёта не совпадает. Обратитесь к администратору.', status=409)
    services.audit(request.user, 'Скачан проверочный отчёт', report.pk)
    response = HttpResponse(data, content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="passengers_{report.pk}.csv"'
    return response


@admin_required
def users(request):
    rows = User.objects.prefetch_related('buses').order_by('username')
    return render(request, 'portal/users.html', {'page_title': 'Пользователи', 'eyebrow': 'Администрирование',
                 'nav': 'users', 'page_obj': Paginator(rows, 25).get_page(request.GET.get('page'))})


@admin_required
def user_create(request):
    form = ConsoleUserCreateForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        with transaction.atomic():
            account = form.save()
            services.audit(request.user, 'Создан пользователь', account.username)
        messages.success(request, 'Пользователь создан.')
        return redirect('users')
    return render(request, 'portal/form.html', {'page_title': 'Новый пользователь', 'eyebrow': 'Администрирование',
                 'nav': 'users', 'form': form, 'submit_label': 'Создать пользователя', 'back_url': 'users'})


@admin_required
def user_edit(request, pk):
    account = get_object_or_404(User, pk=pk)
    form = ConsoleUserEditForm(request.POST or None, instance=account, actor=request.user)
    if request.method == 'POST' and form.is_valid():
        with transaction.atomic():
            account = form.save()
            services.audit(request.user, 'Изменён доступ пользователя', account.username)
        if account.pk == request.user.pk and form.cleaned_data.get('password1'):
            update_session_auth_hash(request, account)
        messages.success(request, 'Изменения сохранены.')
        return redirect('users')
    return render(request, 'portal/form.html', {'page_title': account.username, 'eyebrow': 'Настройки пользователя',
                 'nav': 'users', 'form': form, 'submit_label': 'Сохранить изменения', 'back_url': 'users'})


@admin_required
def buses(request):
    form = BusForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        with transaction.atomic():
            bus = form.save()
            services.audit(request.user, 'Добавлен автобус', bus.bus_id)
        messages.success(request, 'Автобус добавлен. Теперь его можно назначить сотруднику.')
        return redirect('buses')
    rows = Bus.objects.all()  # Registry only: no contact with camera sources.
    return render(request, 'portal/buses.html', {'page_title': 'Автобусы', 'eyebrow': 'Справочники и доступ',
                 'nav': 'buses', 'form': form, 'page_obj': Paginator(rows, 25).get_page(request.GET.get('page'))})


@admin_required
def bus_edit(request, pk):
    bus = get_object_or_404(Bus, pk=pk)
    form = BusForm(request.POST or None, instance=bus)
    if request.method == 'POST' and form.is_valid():
        with transaction.atomic():
            form.save()
            services.audit(request.user, 'Изменена карточка автобуса', bus.bus_id)
        messages.success(request, 'Карточка обновлена.')
        return redirect('buses')
    return render(request, 'portal/form.html', {'page_title': bus.name, 'eyebrow': bus.bus_id,
                 'nav': 'buses', 'form': form, 'submit_label': 'Сохранить карточку', 'back_url': 'buses'})


@admin_required
def audit_log(request):
    return render(request, 'portal/audit.html', {'page_title': 'Журнал действий', 'eyebrow': 'Администрирование',
                 'nav': 'audit', 'page_obj': Paginator(AuditEntry.objects.all(), 30).get_page(request.GET.get('page'))})


def health(request):
    return JsonResponse({'status': 'ok'})
