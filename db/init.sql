CREATE TABLE telemetry (
    unit_id      bigint           NOT NULL,
    received_at  timestamptz      NOT NULL,
    request_id   bigint           NOT NULL,
    device_time  timestamptz,
    lat          double precision,
    lon          double precision,
    speed        integer,
    course       integer,
    altitude     integer,
    satellites   integer,
    fuel_level_l integer,
    engine_rpm   integer,
    engine_temp  integer,
    temperature  integer,
    cells        jsonb            NOT NULL,
    PRIMARY KEY (unit_id, received_at)
);

CREATE INDEX telemetry_received_at_idx ON telemetry (received_at);

CREATE TABLE predictions (
    unit_id       bigint           NOT NULL,
    received_at   timestamptz      NOT NULL,
    score         double precision NOT NULL,
    model_version text             NOT NULL,
    target_stop_id    bigint,
    target_planned_at timestamptz,
    cur_dev_s         double precision,
    created_at    timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (unit_id, received_at),
    FOREIGN KEY (unit_id, received_at) REFERENCES telemetry (unit_id, received_at) ON DELETE CASCADE
);

CREATE ROLE grafana LOGIN PASSWORD 'grafana';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana;
