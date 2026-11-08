"""Opt-in integration tests: TEST_DB_PORT must point at a disposable PostgreSQL."""
from datetime import datetime, timezone
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch

import cv2
import numpy as np
import psycopg2

from config.settings import Settings
from core.counter_engine import PassengerCounterEngine
from core.service import synchronize
from core.service import run
from core.types import CrossingEvent, Detection
from database.db_client import DatabaseClient
from database.local_store import LocalStore


@unittest.skipUnless(os.getenv("TEST_DB_PORT"), "Set TEST_DB_PORT for disposable PostgreSQL tests")
class PostgresTests(unittest.TestCase):
    def setUp(self):
        self.bus = "TEST_" + uuid4().hex[:20]
        self.cfg = Settings(BUS_ID=self.bus)
        self.db = DatabaseClient("127.0.0.1", int(os.environ["TEST_DB_PORT"]),
                                 "bus_analytics", "bus_counter", "test-local-only")
        self.db.initialize()
        self.source_key = uuid4().hex

    def tearDown(self):
        with self.db.get_cursor() as cur:
            for table in ("raw_passenger_events", "daily_passenger_traffic", "current_day_passenger_traffic"):
                cur.execute(f"DELETE FROM {table} WHERE bus_id=%s", (self.bus,))
            cur.execute("DELETE FROM counter_runtime_state WHERE source_key=%s", (self.source_key,))
        self.db.close()

    def event(self, direction="IN", counted=True):
        return CrossingEvent.create(self.cfg, datetime(2026, 11, 6, 22, tzinfo=timezone.utc),
                                    direction, 10, "person", counted,
                                    "COUNTED" if counted else "REPEAT_WITHIN_HOUR").to_dict()

    def test_duplicate_delivery_cannot_increment_twice(self):
        events = [self.event(), self.event("OUT"), self.event(counted=False)]
        self.db.deliver(events)
        self.db.deliver(events)
        with self.db.get_cursor() as cur:
            cur.execute("SELECT total_passengers,total_out,traffic_date::text FROM daily_passenger_traffic WHERE bus_id=%s", (self.bus,))
            self.assertEqual(cur.fetchone(), (1, 1, "2026-11-07"))
            cur.execute("SELECT total_in,total_out FROM current_day_passenger_traffic WHERE bus_id=%s", (self.bus,))
            self.assertEqual(cur.fetchone(), (1, 1))
            cur.execute("SELECT COUNT(*) FROM raw_passenger_events WHERE bus_id=%s", (self.bus,))
            self.assertEqual(cur.fetchone()[0], 3)

    def test_failed_batch_rolls_back_event_and_aggregate(self):
        good, invalid = self.event(), self.event()
        invalid["camera_id"] = "X" * 33
        with self.assertRaises(psycopg2.DataError):
            self.db.deliver([good, invalid])
        with self.db.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM raw_passenger_events WHERE bus_id=%s", (self.bus,))
            self.assertEqual(cur.fetchone()[0], 0)
            cur.execute("SELECT COUNT(*) FROM daily_passenger_traffic WHERE bus_id=%s", (self.bus,))
            self.assertEqual(cur.fetchone()[0], 0)

    def test_checkpoint_and_event_replayed_after_lost_acknowledgement(self):
        with tempfile.TemporaryDirectory() as folder:
            store = LocalStore(Path(folder) / "state.db")
            try:
                event = CrossingEvent(**self.event())
                engine = PassengerCounterEngine(self.cfg)
                engine.day = event.traffic_date
                payload = {"engine": engine.snapshot(), "next_frame": 42, "completed": False,
                           "bus_id": self.bus, "camera_id": self.cfg.CAMERA_ID}
                store.save(self.source_key, payload, [event])
                # Simulate PostgreSQL commit followed by process death before local ack.
                self.db.deliver(store.pending(), store.load())
                self.assertEqual(store.pending_count(), 1)
                synchronize(store, self.db)
                self.assertEqual(store.pending_count(), 0)
                self.assertEqual(self.db.load_checkpoint(self.source_key)["payload"]["next_frame"], 42)
                with self.db.get_cursor() as cur:
                    cur.execute("SELECT total_passengers FROM daily_passenger_traffic WHERE bus_id=%s", (self.bus,))
                    self.assertEqual(cur.fetchone()[0], 1)
            finally:
                store.close()

    def test_checkpoint_cannot_regress(self):
        payload = {"engine": PassengerCounterEngine(self.cfg).snapshot(), "bus_id": self.bus,
                   "camera_id": self.cfg.CAMERA_ID, "next_frame": 100}
        self.db.deliver([], {"source_key": self.source_key, "revision": 5, "payload": payload})
        payload = dict(payload, next_frame=10)
        self.db.deliver([], {"source_key": self.source_key, "revision": 4, "payload": payload})
        self.assertEqual(self.db.load_checkpoint(self.source_key)["payload"]["next_frame"], 100)

    def test_schema_migration_can_run_repeatedly(self):
        self.db.initialized = False
        self.db.initialize()
        self.db.deliver([self.event()])

    def test_server_connection_loss_can_be_retried(self):
        self.db.close()
        self.db.deliver([self.event()])
        with self.db.get_cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            pid = cur.fetchone()[0]
        other = psycopg2.connect(**self.db.params)
        try:
            other.autocommit = True
            with other.cursor() as cur:
                cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
        finally:
            other.close()
        event = self.event("OUT")
        with self.assertRaises((psycopg2.OperationalError, psycopg2.InterfaceError)):
            self.db.deliver([event])
        self.db.deliver([event])

    def _service_config(self, folder):
        path = Path(folder)
        video = path / "crossing.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
        self.assertTrue(writer.isOpened())
        for y in [20, 20, 50, 80, 80, 80]:
            writer.write(np.full((100, 100, 3), y, np.uint8))
        writer.release()
        for name in ("reid.pt", "vest.pt"):
            (path / name).write_bytes(b"test stub: not a real model")
        return replace(self.cfg, VIDEO_SOURCE=str(video), VIDEO_START="2026-11-06T22:00:00+00:00",
                       DB_PORT=int(os.environ["TEST_DB_PORT"]), DB_PASS="test-local-only",
                       STATE_PATH=str(path / "runtime.db"), REID_MODEL_PATH=str(path / "reid.pt"),
                       VEST_MODEL_PATH=str(path / "vest.pt"))

    def test_service_restart_mid_crossing_and_completed_file(self):
        class DetectorStub:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self):
                pass

            def detect(self, frame):
                y = round(float(frame.mean()) / 10) * 10
                return [Detection(1, (40, y - 10, 60, y + 10), 1, np.array([1, 0]), 0)]

        with tempfile.TemporaryDirectory() as folder, patch("core.service.PersonDetector", DetectorStub):
            cfg = self._service_config(folder)
            from core.service import configuration_key
            self.source_key = configuration_key(cfg)
            self.assertEqual(run(cfg, max_frames=3)["in"], 0)
            self.assertEqual(run(cfg)["in"], 1)
            self.assertEqual(run(cfg)["frames"], 0)
            with self.db.get_cursor() as cur:
                cur.execute("SELECT total_passengers FROM daily_passenger_traffic WHERE bus_id=%s", (self.bus,))
                self.assertEqual(cur.fetchone()[0], 1)

    def test_service_keeps_outbox_during_database_outage(self):
        class DetectorStub:
            def __init__(self, *args, **kwargs):
                pass

            def reset(self):
                pass

            def detect(self, frame):
                y = round(float(frame.mean()) / 10) * 10
                return [Detection(1, (40, y - 10, 60, y + 10), 1, np.array([1, 0]), 0)]

        with tempfile.TemporaryDirectory() as folder, patch("core.service.PersonDetector", DetectorStub):
            cfg = self._service_config(folder)
            from core.service import configuration_key
            self.source_key = configuration_key(cfg)
            with patch.object(DatabaseClient, "deliver", side_effect=psycopg2.OperationalError("test outage")):
                self.assertEqual(run(cfg)["in"], 1)
            store = LocalStore(cfg.STATE_PATH)
            try:
                self.assertEqual(store.pending_count(), 1)
            finally:
                store.close()
            self.assertEqual(run(cfg)["frames"], 0)
            with self.db.get_cursor() as cur:
                cur.execute("SELECT total_passengers FROM daily_passenger_traffic WHERE bus_id=%s", (self.bus,))
                self.assertEqual(cur.fetchone()[0], 1)
