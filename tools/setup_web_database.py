"""Provision a dedicated SELECT-only role, without changing counter data."""
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

MARKER = 'Passenger console analytics reader'


def provision():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env.web', override=False)
    load_dotenv(root / '.env', override=False)
    reader = os.environ.get('WEB_ANALYTICS_DB_USER', 'bus_dashboard')
    password = os.environ.get('WEB_ANALYTICS_DB_PASS', '')
    admin = os.environ.get('DB_USER', 'bus_counter')
    dbname = os.environ.get('DB_NAME', 'bus_analytics')
    if (not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', reader) or reader == admin
            or len(password) < 24 or password == 'generate-a-readonly-database-password'):
        raise ValueError('Configure a separate reader and generated password in .env.web')
    if os.environ.get('WEB_ANALYTICS_DB_NAME', dbname) != dbname:
        raise ValueError('WEB_ANALYTICS_DB_NAME must match DB_NAME')
    conn = psycopg2.connect(host=os.environ.get('DB_HOST', '127.0.0.1'),
                           port=int(os.environ.get('DB_PORT', '5432')), dbname=dbname, user=admin,
                           password=os.environ.get('DB_PASS', ''), connect_timeout=5,
                           options='-c statement_timeout=10000 -c lock_timeout=3000')
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("SELECT r.oid, r.rolsuper, r.rolcreatedb, r.rolcreaterole, r.rolreplication, r.rolbypassrls, shobj_description(r.oid, 'pg_authid') FROM pg_roles r WHERE rolname=%s", (reader,))
            existing = cursor.fetchone()
            if existing:
                if any(existing[1:6]) or existing[6] != MARKER:
                    raise ValueError('Existing reader role is not managed by the console; choose another name')
                cursor.execute('SELECT 1 FROM pg_auth_members WHERE member=%s LIMIT 1', (existing[0],))
                if cursor.fetchone():
                    raise ValueError('Reader has unexpected role memberships')
            role = sql.Identifier(reader)
            if not existing:
                cursor.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT PASSWORD %s').format(role), (password,))
                cursor.execute(sql.SQL('COMMENT ON ROLE {} IS %s').format(role), (MARKER,))
            else:
                cursor.execute(sql.SQL('ALTER ROLE {} LOGIN PASSWORD %s').format(role), (password,))
            cursor.execute(sql.SQL('REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {}').format(role))
            cursor.execute(sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(sql.Identifier(dbname), role))
            cursor.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(role))
            for table in ['daily_passenger_traffic', 'current_day_passenger_traffic', 'raw_passenger_events']:
                cursor.execute(sql.SQL('GRANT SELECT ON {} TO {}').format(sql.Identifier(table), role))
            cursor.execute(sql.SQL('ALTER ROLE {} SET default_transaction_read_only = on').format(role))
            cursor.execute(sql.SQL('ALTER ROLE {} SET statement_timeout = %s').format(role), ('3000',))
        print('Read-only analytics role ready. Runtime checkpoints and passenger data were not modified.')
    finally:
        conn.close()


if __name__ == '__main__':
    try:
        provision()
    except psycopg2.Error as error:
        raise SystemExit(f'Database setup failed (SQLSTATE {error.pgcode or "connection"}); check configuration. Secrets were not printed.') from None
