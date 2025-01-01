BEGIN;
    DROP TABLE IF EXISTS users_staging;
    CREATE TABLE users_staging (LIKE users INCLUDING ALL);
    \copy users_staging FROM '../../data/replace_users/users.csv' CSV HEADER
    LOCK TABLE users IN ACCESS EXCLUSIVE MODE;
    ALTER TABLE users RENAME TO users_old;
    ALTER TABLE users_staging RENAME TO users;
    DROP TABLE users_old;
COMMIT;
