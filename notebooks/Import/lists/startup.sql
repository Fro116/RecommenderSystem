DROP TABLE IF EXISTS collect_users_staging;

CREATE TABLE collect_users_staging (
    source TEXT,
    username TEXT,
    userid BIGINT,
    fingerprint TEXT,
    data BYTEA,
    db_refreshed_at DOUBLE PRECISION
);
