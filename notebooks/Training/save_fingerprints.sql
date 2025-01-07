DROP TABLE IF EXISTS collect_users_staging;
CREATE TABLE collect_users_staging (LIKE collect_users INCLUDING ALL);
\copy collect_users_staging FROM PROGRAM 'zstd -cdq ../../data/users/animeplanet.fingerprints.csv.zst' CSV HEADER;
\copy collect_users_staging FROM PROGRAM 'zstd -cdq ../../data/users/kitsu.fingerprints.csv.zst' CSV HEADER;
\copy collect_users_staging FROM PROGRAM 'zstd -cdq ../../data/users/anilist.fingerprints.csv.zst' CSV HEADER;
\copy collect_users_staging FROM PROGRAM 'zstd -cdq ../../data/users/mal.fingerprints.csv.zst' CSV HEADER;

BEGIN;
    LOCK TABLE collect_users IN ACCESS EXCLUSIVE MODE;
    ALTER TABLE collect_users RENAME TO collect_users_old;
    ALTER TABLE collect_users_staging RENAME TO collect_users;
    DROP TABLE collect_users_old;
COMMIT;

BEGIN;
    DELETE FROM inference_users WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;
COMMIT;
