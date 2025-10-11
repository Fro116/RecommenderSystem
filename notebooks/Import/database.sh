#!/bin/bash
set -euxo pipefail
secretdir=$1
connstr=`head -n 1 $secretdir/db.inference.txt | tr -d '\n'`

# fn="item_autocomplete.csv"
# rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
# unzstd $fn.zstd
# rm $fn.zstd
# db="autocomplete_items"
# fn=`pwd`/$fn
# psql "$connstr" <<EOF
# DROP TABLE IF EXISTS ${db}_staging;
# CREATE TABLE ${db}_staging (LIKE $db);
# \copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
# CREATE UNIQUE INDEX ${db}_staging_medium_prefix_idx ON ${db}_staging (medium, prefix);
# BEGIN;
# LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
# ALTER TABLE ${db} RENAME TO ${db}_old;
# ALTER TABLE ${db}_staging RENAME TO ${db};
# DROP TABLE ${db}_old;
# ALTER INDEX ${db}_staging_medium_prefix_idx RENAME TO ${db}_medium_prefix_idx;
# COMMIT;
# EOF
# rm $fn

# fn="user_autocomplete.csv"
# rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
# unzstd $fn.zstd
# rm $fn.zstd
# db="autocomplete_users"
# fn=`pwd`/$fn
# psql "$connstr" <<EOF
# DROP TABLE IF EXISTS ${db}_staging;
# CREATE TABLE ${db}_staging (LIKE $db);
# \copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
# CREATE UNIQUE INDEX ${db}_staging_source_prefix_idx ON ${db}_staging (source, prefix);
# BEGIN;
# LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
# ALTER TABLE ${db} RENAME TO ${db}_old;
# ALTER TABLE ${db}_staging RENAME TO ${db};
# DROP TABLE ${db}_old;
# ALTER INDEX ${db}_staging_source_prefix_idx RENAME TO ${db}_source_prefix_idx;
# COMMIT;
# EOF
# rm $fn

# fn="user_histories.csv"
# rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
# unzstd $fn.zstd
# rm $fn.zstd
# db="user_histories"
# fn=`pwd`/$fn
# psql "$connstr" <<EOF
# DROP TABLE IF EXISTS ${db}_staging;
# CREATE TABLE ${db}_staging (LIKE $db);
# \copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
# CREATE INDEX ${db}_staging_source_lower_username_idx ON ${db}_staging (source, lower(username));
# BEGIN;
# LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
# ALTER TABLE ${db} RENAME TO ${db}_old;
# ALTER TABLE ${db}_staging RENAME TO ${db};
# DROP TABLE ${db}_old;
# ALTER INDEX ${db}_staging_source_lower_username_idx RENAME TO ${db}_source_lower_username_idx;
# COMMIT;
# DELETE FROM online_user_histories WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;
# EOF
# rm $fn

bucket=`cat $secretdir/gcp.bucket.txt`
connstr=`tail -n 1 $secretdir/db.inference.txt | tr -d '\n'`
gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json

# df="item_autocomplete"
# db="autocomplete_items"
# rclone copyto -Pv r2:rsys/database/import/$df.csv.zstd $df.csv.zstd
# unzstd $df.csv.zstd
# rm $df.csv.zstd
# gcloud storage cp $df.csv $bucket/
# psql "$connstr" -c "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (medium INT, prefix TEXT, data BYTEA);"
# gcloud sql import csv inference $bucket/$df.csv --database=postgres --table=${db}_staging -q --async
# sleep 10
# PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
# gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited
# gcloud storage rm $bucket/$df.csv
# psql "$connstr" -c "CREATE UNIQUE INDEX ${db}_staging_medium_prefix_idx ON ${db}_staging (medium, prefix);"
# psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_medium_prefix_idx RENAME TO ${db}_medium_prefix_idx; COMMIT;"
# rm $df.csv

# df="user_autocomplete"
# db="autocomplete_users"
# rclone copyto -Pv r2:rsys/database/import/$df.csv.zstd $df.csv.zstd
# unzstd $df.csv.zstd
# rm $df.csv.zstd
# gcloud storage cp $df.csv $bucket/
# psql "$connstr" -c "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (source TEXT, prefix TEXT, data BYTEA);"
# gcloud sql import csv inference $bucket/$df.csv --database=postgres --table=${db}_staging -q --async
# sleep 10
# PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
# gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited
# gcloud storage rm $bucket/$df.csv
# psql "$connstr" -c "CREATE UNIQUE INDEX ${db}_staging_source_prefix_idx ON ${db}_staging (source, prefix);"
# psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_source_prefix_idx RENAME TO ${db}_source_prefix_idx; COMMIT;"
# rm $df.csv

df="user_histories"
db="user_histories"
rclone copyto -Pv r2:rsys/database/import/$df.csv.zstd $df.csv.zstd
unzstd $df.csv.zstd
rm $df.csv.zstd
tail -n +2 $df.csv > $df.csv.headerless
mv $df.csv.headerless $df.csv
gcloud storage cp $df.csv $bucket/
psql "$connstr" -c  "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (source TEXT, username TEXT, userid BIGINT, data BYTEA, db_refreshed_at DOUBLE PRECISION);"
gcloud sql import csv inference $bucket/$df.csv --database=postgres --table=${db}_staging -q --async
sleep 10
PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited
psql "$connstr" -c "CREATE INDEX ${db}_staging_source_lower_username_idx ON ${db}_staging (source, lower(username));"
psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_source_lower_username_idx RENAME TO ${db}_source_lower_username_idx; COMMIT;"
psql "$connstr" -c "DELETE FROM online_user_histories WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;"
gcloud storage rm $bucket/$df.csv
rm $df.csv
