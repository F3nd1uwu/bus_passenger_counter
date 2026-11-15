"""Bounded, scoped SELECT queries; never read runtime identity checkpoints."""
import csv
import io

from django.db.models import Count, Max, Q, Sum

from web.analytics.models import CameraDaily, PassengerEvent
from .models import AuditEntry, Bus


def audit(user, action, target=''):
    AuditEntry.objects.create(actor=user, actor_name=user.username, action=action, target=str(target)[:150])


def registry():
    return {bus.bus_id: bus for bus in Bus.objects.all()}


def bus_label(bus_id, buses):
    bus = buses.get(bus_id)
    if bus:
        return {'name': bus.name, 'mode': bus.mode, 'mode_label': bus.get_mode_display()}
    mode = 'test' if bus_id.startswith('TEST_') else 'unknown'
    return {'name': bus_id, 'mode': mode, 'mode_label': 'Тестовый' if mode == 'test' else 'Не проверен'}


def scoped(model, user, filters):
    queryset = model.objects.using('analytics').filter(traffic_date__range=(filters['date_from'], filters['date_to']))
    if not user.is_console_admin:
        queryset = queryset.filter(bus_id__in=list(user.buses.values_list('bus_id', flat=True)))
    if filters.get('bus'):
        queryset = queryset.filter(bus_id=filters['bus'])
    mode = filters.get('mode', 'all')
    if mode != 'all':
        known = list(Bus.objects.values_list('bus_id', flat=True))
        named = list(Bus.objects.filter(mode=mode).values_list('bus_id', flat=True))
        legacy_test = Q(bus_id__startswith='TEST_') & ~Q(bus_id__in=known)
        if mode == 'test':
            queryset = queryset.filter(Q(bus_id__in=named) | legacy_test)
        elif mode == 'production':
            queryset = queryset.filter(bus_id__in=named)
        else:
            queryset = queryset.filter(Q(bus_id__in=named) | (~Q(bus_id__in=known) & ~Q(bus_id__startswith='TEST_')))
    return queryset


def report_rows(user, filters):
    return scoped(CameraDaily, user, filters).values('traffic_date', 'bus_id').annotate(
        counted_in=Sum('total_in'), counted_out=Sum('total_out'), last_update=Max('updated_at')).order_by('-traffic_date', 'bus_id')


def decorate(rows, buses):
    return [{**row, **bus_label(row['bus_id'], buses)} for row in rows]


def overview(user, filters):
    cameras = scoped(CameraDaily, user, filters)
    totals = cameras.aggregate(counted_in=Sum('total_in'), counted_out=Sum('total_out'), last_update=Max('updated_at'))
    totals['counted_in'] = totals['counted_in'] or 0
    totals['counted_out'] = totals['counted_out'] or 0
    totals['buses'] = cameras.values('bus_id').distinct().count()
    totals['cameras'] = cameras.values('bus_id', 'camera_id').distinct().count()
    days = list(cameras.values('traffic_date').annotate(counted_in=Sum('total_in'), counted_out=Sum('total_out')).order_by('traffic_date'))
    maximum = max([max(day['counted_in'], day['counted_out']) for day in days], default=1) or 1
    for day in days:
        day['in_height'] = max(1, round(100 * day['counted_in'] / maximum))
        day['out_height'] = max(1, round(100 * day['counted_out'] / maximum))
        day['in_y'] = 120 - day['in_height']
        day['out_y'] = 120 - day['out_height']
    totals['chart_max'] = maximum
    recent = cameras.values('traffic_date', 'bus_id', 'camera_id', 'total_in', 'total_out', 'updated_at').order_by('-updated_at')[:8]
    return totals, days, list(recent)


def event_summary(user, filters):
    result = scoped(PassengerEvent, user, filters).aggregate(
        accepted_events=Count('id', filter=Q(counted=True)), excluded_events=Count('id', filter=Q(counted=False)))
    return {'counted': result['accepted_events'], 'excluded': result['excluded_events']}


def csv_bytes(rows):
    def safe(value):
        value = str(value)
        return "'" + value if value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value
    output = io.StringIO(newline='')
    writer = csv.writer(output, delimiter=';', lineterminator='\r\n')
    writer.writerow(['Дата', 'Код автобуса', 'Автобус', 'Тип данных', 'Учтённые входы', 'Учтённые выходы', 'Обновлено'])
    for row in rows:
        writer.writerow([safe(row[key]) for key in ['date', 'bus_id', 'name', 'mode_label', 'counted_in', 'counted_out', 'updated_at']])
    return ('\ufeff' + output.getvalue()).encode('utf-8')
