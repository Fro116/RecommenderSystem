#!/bin/bash
set -euxo pipefail
secretdir=$1
connstr=`head -n 1 $secretdir/db.inference.txt | tr -d '\n'`

fn="item_autocomplete.csv"
rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
unzstd $fn.zstd
rm $fn.zstd
db="autocomplete_items"
fn=`pwd`/$fn
psql "$connstr" <<EOF
DROP TABLE IF EXISTS ${db}_staging;
CREATE TABLE ${db}_staging (LIKE $db);
\copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
CREATE UNIQUE INDEX ${db}_staging_medium_prefix_idx ON ${db}_staging (medium, prefix);
BEGIN;
LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
ALTER TABLE ${db} RENAME TO ${db}_old;
ALTER TABLE ${db}_staging RENAME TO ${db};
DROP TABLE ${db}_old;
ALTER INDEX ${db}_staging_medium_prefix_idx RENAME TO ${db}_medium_prefix_idx;
COMMIT;
EOF
rm $fn

fn="user_autocomplete.csv"
rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
unzstd $fn.zstd
rm $fn.zstd
db="autocomplete_users"
fn=`pwd`/$fn
psql "$connstr" <<EOF
DROP TABLE IF EXISTS ${db}_staging;
CREATE TABLE ${db}_staging (LIKE $db);
\copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
CREATE UNIQUE INDEX ${db}_staging_source_prefix_idx ON ${db}_staging (source, prefix);
BEGIN;
LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
ALTER TABLE ${db} RENAME TO ${db}_old;
ALTER TABLE ${db}_staging RENAME TO ${db};
DROP TABLE ${db}_old;
ALTER INDEX ${db}_staging_source_prefix_idx RENAME TO ${db}_source_prefix_idx;
COMMIT;
EOF
rm $fn

fn="user_histories.csv"
rclone copyto -Pv r2:rsys/database/import/$fn.zstd $fn.zstd
unzstd $fn.zstd
rm $fn.zstd
db="user_histories"
fn=`pwd`/$fn
psql "$connstr" <<EOF
DROP TABLE IF EXISTS ${db}_staging;
CREATE TABLE ${db}_staging (LIKE $db);
\copy ${db}_staging FROM '$fn' WITH (FORMAT csv, HEADER true);
CREATE INDEX ${db}_staging_source_lower_username_idx ON ${db}_staging (source, lower(username));
BEGIN;
LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
ALTER TABLE ${db} RENAME TO ${db}_old;
ALTER TABLE ${db}_staging RENAME TO ${db};
DROP TABLE ${db}_old;
ALTER INDEX ${db}_staging_source_lower_username_idx RENAME TO ${db}_source_lower_username_idx;
COMMIT;
DELETE FROM online_user_histories WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;
EOF
rm $fn