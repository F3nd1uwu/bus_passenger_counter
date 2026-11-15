from django.db import models


class CameraDaily(models.Model):
    pk = models.CompositePrimaryKey('traffic_date', 'bus_id', 'camera_id')
    traffic_date = models.DateField()
    bus_id = models.CharField(max_length=32)
    camera_id = models.CharField(max_length=32)
    total_in = models.BigIntegerField()
    total_out = models.BigIntegerField()
    updated_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'current_day_passenger_traffic'


class PassengerEvent(models.Model):
    id = models.BigAutoField(primary_key=True)
    event_id = models.UUIDField(null=True)
    bus_id = models.CharField(max_length=32)
    camera_id = models.CharField(max_length=32)
    detected_at = models.DateTimeField()
    traffic_date = models.DateField(null=True)
    event_type = models.CharField(max_length=16)
    track_id = models.BigIntegerField()
    identity_id = models.TextField(null=True)
    counted = models.BooleanField()
    reason = models.TextField(null=True)

    class Meta:
        managed = False
        db_table = 'raw_passenger_events'
