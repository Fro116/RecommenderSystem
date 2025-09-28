#!/bin/bash
set -euxo pipefail
cd ../../../data/import/lists
df="new_histories.csv"
db="user_histories"
secretdir="../../../secrets"
bucket=`cat $secretdir/gcp.bucket.txt`
connstr=`head -n 1 $secretdir/db.inference.txt | tr -d '\n'`

tail -n +2 $df > $df.headerless
mv $df.headerless $df

gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json
gcloud storage cp $df $bucket/
psql "$connstr" -c  "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (source TEXT, username TEXT, userid BIGINT, data BYTEA, db_refreshed_at DOUBLE PRECISION);"
gcloud sql import csv inference $bucket/${df} --database=postgres --table=${db}_staging -q --async
sleep 10
PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited

psql "$connstr" -c "CREATE INDEX ${db}_staging_source_lower_username_idx ON ${db}_staging (source, lower(username));"
psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_source_lower_username_idx RENAME TO ${db}_source_lower_username_idx; COMMIT;"
psql "$connstr" -c "DELETE FROM online_user_histories WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;"
gcloud storage rm $bucket/$df
