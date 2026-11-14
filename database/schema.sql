-- Additive migration: old event rows and IN totals are retained.
CREATE TABLE IF NOT EXISTS raw_passenger_events (
    id BIGSERIAL PRIMARY KEY,
    bus_id VARCHAR(32) NOT NULL,
    camera_id VARCHAR(32) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(16) NOT NULL CHECK (event_type IN ('IN', 'OUT', 'BOUNCE')),
    track_id BIGINT NOT NULL,
    reid_features BYTEA
);
ALTER TABLE raw_passenger_events ADD COLUMN IF NOT EXISTS event_id UUID;
ALTER TABLE raw_passenger_events ADD COLUMN IF NOT EXISTS identity_id TEXT;
ALTER TABLE raw_passenger_events ADD COLUMN IF NOT EXISTS traffic_date DATE;
ALTER TABLE raw_passenger_events ADD COLUMN IF NOT EXISTS counted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE raw_passenger_events ADD COLUMN IF NOT EXISTS reason TEXT;
ALTER TABLE raw_passenger_events ALTER COLUMN track_id TYPE BIGINT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_uuid ON raw_passenger_events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_time_bus ON raw_passenger_events(detected_at, bus_id);
CREATE INDEX IF NOT EXISTS idx_events_day ON raw_passenger_events(traffic_date, bus_id);

CREATE TABLE IF NOT EXISTS daily_passenger_traffic (
    traffic_date DATE NOT NULL,
    bus_id VARCHAR(32) NOT NULL,
    total_passengers BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (traffic_date, bus_id)
);
ALTER TABLE daily_passenger_traffic ADD COLUMN IF NOT EXISTS total_out BIGINT NOT NULL DEFAULT 0;
ALTER TABLE daily_passenger_traffic ALTER COLUMN total_passengers TYPE BIGINT;

-- Day-by-day live counters per camera, also useful for reconciliation.
CREATE TABLE IF NOT EXISTS current_day_passenger_traffic (
    traffic_date DATE NOT NULL,
    bus_id VARCHAR(32) NOT NULL,
    camera_id VARCHAR(32) NOT NULL,
    total_in BIGINT NOT NULL DEFAULT 0,
    total_out BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (traffic_date, bus_id, camera_id)
);

-- Contains identities, hourly timers and the video cursor, not only totals.
CREATE TABLE IF NOT EXISTS counter_runtime_state (
    source_key TEXT PRIMARY KEY,
    revision BIGINT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
