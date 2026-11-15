"""Populate isolated, explicitly labelled demo analytics only."""
from datetime import datetime, time, timedelta
import os
from uuid import uuid4

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction
from django.utils import timezone

from web.analytics.models import CameraDaily, PassengerEvent
from web.portal.models import Bus, User


class Command(BaseCommand):
    help = 'Create synthetic analytics in WEB_DEMO=1. Never connects to production PostgreSQL.'

    def handle(self, *args, **options):
        if not settings.DEMO or connections['analytics'].vendor != 'sqlite':
            raise CommandError('Demo seeding requires WEB_DEMO=1 and its isolated SQLite source')
        password = os.environ.get('WEB_DEMO_PASSWORD', '')
        if len(password) < 12:
            raise CommandError('Set WEB_DEMO_PASSWORD to a separate local demo password (at least 12 characters)')
        connection = connections['analytics']
        existing = connection.introspection.table_names()
        with connection.schema_editor() as editor:
            for model in [CameraDaily, PassengerEvent]:
                if model._meta.db_table not in existing:
                    editor.create_model(model)
        admin, _ = User.objects.get_or_create(username='demo-admin', defaults={'role': User.Role.ADMIN, 'first_name': 'Александр', 'last_name': 'Морозов'})
        employee, _ = User.objects.get_or_create(username='demo-employee', defaults={'role': User.Role.EMPLOYEE, 'first_name': 'Анна', 'last_name': 'Волкова'})
        for user in [admin, employee]:
            user.set_password(password)
            user.save()
        records = [('DEMO_BUS_041', 'Автобус 041', Bus.Mode.PRODUCTION),
                   ('DEMO_BUS_107', 'Автобус 107', Bus.Mode.PRODUCTION),
                   ('TEST_VIDEO_1', 'Проверка · видео 1', Bus.Mode.TEST),
                   ('TEST_VIDEO_2', 'Проверка · видео 2', Bus.Mode.TEST)]
        for bus_id, name, mode in records:
            bus, _ = Bus.objects.get_or_create(bus_id=bus_id, defaults={'name': name, 'mode': mode})
            if bus_id in {'DEMO_BUS_041', 'TEST_VIDEO_1'}:
                bus.users.add(employee)
        today = timezone.localdate()
        # This is a synthetic dataset, not an accuracy or passenger-flow claim.
        with transaction.atomic(using='analytics'), connection.cursor() as cursor:
            cursor.execute('DELETE FROM current_day_passenger_traffic')
            cursor.execute('DELETE FROM raw_passenger_events')
            for offset in range(7):
                day = today - timedelta(days=6-offset)
                for index, (bus_id, _, _) in enumerate(records):
                    incoming = (124 + index * 37 + offset * 19) if index < 2 else (31 + index * 9 + offset * 4)
                    outgoing = max(0, incoming - 5 + offset % 3)
                    stamp = timezone.make_aware(datetime.combine(day, time(17, 40)))
                    cursor.execute('INSERT INTO current_day_passenger_traffic (traffic_date,bus_id,camera_id,total_in,total_out,updated_at) VALUES (%s,%s,%s,%s,%s,%s)',
                                   [day, bus_id, 'DOOR_FRONT', incoming, outgoing, stamp])
                    # Full event rows reconcile with aggregate counts in the demo.
                    for direction, amount in [('IN', incoming), ('OUT', outgoing)]:
                        for number in range(amount):
                            event_at = timezone.make_aware(datetime.combine(day, time(6, 0))) + timedelta(seconds=number * 80 + index*13)
                            cursor.execute('INSERT INTO raw_passenger_events (event_id,bus_id,camera_id,detected_at,traffic_date,event_type,track_id,identity_id,counted,reason) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                                           [uuid4().hex, bus_id, 'DOOR_FRONT', event_at, day, direction, number + 1,
                                            'demo-person-' + str(number), True, 'COUNTED'])
                    for number, reason in enumerate(['REPEAT_WITHIN_HOUR', 'STAFF', 'INSUFFICIENT_MODEL_EVIDENCE']):
                        cursor.execute('INSERT INTO raw_passenger_events (event_id,bus_id,camera_id,detected_at,traffic_date,event_type,track_id,identity_id,counted,reason) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                                       [uuid4().hex, bus_id, 'DOOR_FRONT', stamp, day, 'IN', 9000 + number,
                                        'demo-excluded', False, reason])
        self.stdout.write('Isolated synthetic demo prepared: demo-admin and demo-employee. Password was not printed.')
