CREATE INDEX collect_users_staging_source_username_idx ON collect_users_staging (source, lower(username));

BEGIN;
    LOCK TABLE collect_users IN ACCESS EXCLUSIVE MODE;
    ALTER TABLE collect_users RENAME TO collect_users_old;
    ALTER TABLE collect_users_staging RENAME TO collect_users;
    DROP TABLE collect_users_old;
    ALTER INDEX collect_users_staging_source_username_idx RENAME TO collect_users_source_username_idx;
COMMIT;

BEGIN;
    DELETE FROM inference_users WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;
COMMIT;

CREATE TABLE collect_users_staging (
    source TEXT,
    username TEXT,
    userid BIGINT,
    fingerprint TEXT,
    data BYTEA,
    db_refreshed_at DOUBLE PRECISION
);
