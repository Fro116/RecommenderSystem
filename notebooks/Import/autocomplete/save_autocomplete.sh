#!/bin/bash
set -euxo pipefail
cd "../../../data/import/autocomplete"
df="user_autocomplete"
db="autocomplete_users"
secretdir="../../../secrets"
bucket=`cat $secretdir/gcp.bucket.txt`
connstr=`head -n 1 $secretdir/db.inference.txt | tr -d '\n'`

mlr --csv cat $df.*.csv > $df.csv
rm $df.*.csv
gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json
gcloud storage cp $df.csv $bucket/

psql "$connstr" -c "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (source TEXT, prefix TEXT, data BYTEA);"
gcloud sql import csv inference $bucket/$df.csv --database=postgres --table=${db}_staging -q --async
sleep 10
PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited
gcloud storage rm $bucket/$df.csv

psql "$connstr" -c "CREATE UNIQUE INDEX ${db}_staging_source_prefix_idx ON ${db}_staging (source, prefix);"
psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_source_prefix_idx RENAME TO ${db}_source_prefix_idx; COMMIT;"
