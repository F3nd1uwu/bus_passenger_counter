"""Idempotent PostgreSQL delivery; events and aggregates share a transaction."""
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.extras import Json


class DatabaseClient:
    def __init__(self, host, port, dbname, user, password, sslmode="prefer"):
        self.params = dict(host=host, port=port, dbname=dbname, user=user,
                           password=password, sslmode=sslmode, connect_timeout=3,
                           options="-c statement_timeout=5000 -c lock_timeout=3000")
        self.conn = None
        self.initialized = False

    @contextmanager
    def get_cursor(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.params)
        try:
            with self.conn:
                with self.conn.cursor() as cur:
                    yield cur
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            self.close()
            raise

    def initialize(self):
        if not self.initialized:
            with self.get_cursor() as cur:
                cur.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
            self.initialized = True

    def load_checkpoint(self, source_key):
        self.initialize()
        with self.get_cursor() as cur:
            cur.execute("SELECT revision, payload FROM counter_runtime_state WHERE source_key=%s", (source_key,))
            row = cur.fetchone()
        return {"revision": row[0], "source_key": source_key, "payload": row[1]} if row else None

    def deliver(self, events, checkpoint=None):
        self.initialize()
        with self.get_cursor() as cur:
            for event in events:
                if event["event_type"] not in {"IN", "OUT"}:
                    raise ValueError("Unsupported event type")
                cur.execute(
                    """INSERT INTO raw_passenger_events
                       (event_id,bus_id,camera_id,detected_at,traffic_date,event_type,track_id,identity_id,counted,reason)
                       VALUES (%(event_id)s,%(bus_id)s,%(camera_id)s,%(detected_at)s,%(traffic_date)s,
                               %(event_type)s,%(track_id)s,%(identity_id)s,%(counted)s,%(reason)s)
                       ON CONFLICT (event_id) DO NOTHING RETURNING event_id""", event)
                inserted = cur.fetchone() is not None
                if inserted and event["counted"]:
                    inc_in = int(event["event_type"] == "IN")
                    inc_out = int(event["event_type"] == "OUT")
                    cur.execute(
                        """INSERT INTO daily_passenger_traffic
                           (traffic_date,bus_id,total_passengers,total_out) VALUES (%s,%s,%s,%s)
                           ON CONFLICT (traffic_date,bus_id) DO UPDATE SET
                           total_passengers=daily_passenger_traffic.total_passengers+EXCLUDED.total_passengers,
                           total_out=daily_passenger_traffic.total_out+EXCLUDED.total_out,
                           updated_at=CURRENT_TIMESTAMP""",
                        (event["traffic_date"], event["bus_id"], inc_in, inc_out))
                    cur.execute(
                        """INSERT INTO current_day_passenger_traffic
                           (traffic_date,bus_id,camera_id,total_in,total_out) VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT (traffic_date,bus_id,camera_id) DO UPDATE SET
                           total_in=current_day_passenger_traffic.total_in+EXCLUDED.total_in,
                           total_out=current_day_passenger_traffic.total_out+EXCLUDED.total_out,
                           updated_at=CURRENT_TIMESTAMP""",
                        (event["traffic_date"], event["bus_id"], event["camera_id"], inc_in, inc_out))
            if checkpoint:
                cur.execute(
                    """INSERT INTO counter_runtime_state (source_key,revision,payload) VALUES (%s,%s,%s)
                       ON CONFLICT (source_key) DO UPDATE SET revision=EXCLUDED.revision,
                       payload=EXCLUDED.payload,updated_at=CURRENT_TIMESTAMP
                       WHERE counter_runtime_state.revision <= EXCLUDED.revision""",
                    (checkpoint["source_key"], checkpoint["revision"], Json(checkpoint["payload"])))
                payload = checkpoint["payload"]
                day = payload["engine"]["day"]
                if day:
                    for query in (
                        """INSERT INTO daily_passenger_traffic (traffic_date,bus_id) VALUES (%s,%s)
                           ON CONFLICT DO NOTHING""",
                        """INSERT INTO current_day_passenger_traffic (traffic_date,bus_id,camera_id)
                           VALUES (%s,%s,%s) ON CONFLICT (traffic_date,bus_id,camera_id)
                           DO UPDATE SET updated_at=CURRENT_TIMESTAMP""",
                    ):
                        params = (day, payload["bus_id"], payload["camera_id"])
                        cur.execute(query, params if "camera_id" in query else params[:2])

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
