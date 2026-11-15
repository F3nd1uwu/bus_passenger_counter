"""Opt-in SELECT-only role tests on disposable PostgreSQL, never a live database."""
import os
import secrets
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg2
from psycopg2 import sql

from database.db_client import DatabaseClient


@unittest.skipUnless(os.getenv('TEST_DB_PORT'), 'Set TEST_DB_PORT for disposable PostgreSQL tests')
class WebDatabaseTests(unittest.TestCase):
    def setUp(self):
        from tools.setup_web_database import provision
        self.provision = provision
        self.reader = 'qa_web_' + uuid4().hex[:16]
        self.password = secrets.token_urlsafe(32)
        self.env = patch.dict(os.environ, {
            'DB_HOST': '127.0.0.1', 'DB_PORT': os.environ['TEST_DB_PORT'],
            'DB_USER': 'bus_counter', 'DB_NAME': 'bus_analytics', 'DB_PASS': 'test-local-only',
            'WEB_ANALYTICS_DB_USER': self.reader, 'WEB_ANALYTICS_DB_PASS': self.password,
            'WEB_ANALYTICS_DB_NAME': 'bus_analytics',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        db = DatabaseClient('127.0.0.1', int(os.environ['TEST_DB_PORT']),
                            'bus_analytics', 'bus_counter', 'test-local-only')
        try:
            db.initialize()
        finally:
            db.close()
        self.admin = psycopg2.connect(host='127.0.0.1', port=os.environ['TEST_DB_PORT'],
            dbname='bus_analytics', user='bus_counter', password='test-local-only')
        self.addCleanup(self.admin.close)
        self.addCleanup(self.remove_test_role)

    def remove_test_role(self):
        self.admin.rollback()
        with self.admin, self.admin.cursor() as cur:
            cur.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (self.reader,))
            if cur.fetchone():
                cur.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(self.reader)))
                cur.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(self.reader)))

    def connect_reader(self):
        return psycopg2.connect(host='127.0.0.1', port=os.environ['TEST_DB_PORT'],
            dbname='bus_analytics', user=self.reader, password=self.password)

    def test_reader_can_select_statistics_but_cannot_read_runtime_or_write(self):
        self.provision()
        conn = self.connect_reader()
        try:
            for table in ['raw_passenger_events', 'daily_passenger_traffic', 'current_day_passenger_traffic']:
                with conn, conn.cursor() as cur:
                    cur.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table)))
                    self.assertGreaterEqual(cur.fetchone()[0], 0)
            with self.assertRaises(psycopg2.errors.InsufficientPrivilege):
                with conn, conn.cursor() as cur:
                    cur.execute('SELECT payload FROM counter_runtime_state LIMIT 1')
            # Bypass the connection's read-only default: table privileges must still prevent writes.
            with self.assertRaises(psycopg2.errors.InsufficientPrivilege):
                with conn, conn.cursor() as cur:
                    cur.execute('SET TRANSACTION READ WRITE')
                    cur.execute('DELETE FROM raw_passenger_events WHERE false')
            with conn, conn.cursor() as cur:
                cur.execute('SHOW default_transaction_read_only')
                self.assertEqual(cur.fetchone()[0], 'on')
        finally:
            conn.close()

    def test_setup_is_repeatable_and_does_not_change_worker_or_tables(self):
        def schema():
            with self.admin, self.admin.cursor() as cur:
                cur.execute("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema='public' ORDER BY 1,2")
                return cur.fetchall()
        before = schema()
        self.provision()
        self.provision()
        self.assertEqual(before, schema())
        with self.admin, self.admin.cursor() as cur:
            cur.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s', (self.reader,))
            self.assertEqual(cur.fetchone(), (False,) * 5)
            cur.execute('SELECT current_user')
            self.assertEqual(cur.fetchone()[0], 'bus_counter')

    def test_unmanaged_existing_role_is_preserved_and_rejected(self):
        with self.admin, self.admin.cursor() as cur:
            cur.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(sql.Identifier(self.reader)))
        with self.assertRaisesRegex(ValueError, 'not managed'):
            self.provision()
        with self.admin, self.admin.cursor() as cur:
            cur.execute('SELECT rolcanlogin FROM pg_roles WHERE rolname=%s', (self.reader,))
            self.assertFalse(cur.fetchone()[0])
