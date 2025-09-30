#!/bin/bash
set -euxo pipefail
cd "../../../data/import/autocomplete_items"
df="item_autocomplete"
db="autocomplete_items"
secretdir="../../../secrets"
bucket=`cat $secretdir/gcp.bucket.txt`
connstr=`tail -n 1 $secretdir/db.inference.txt | tr -d '\n'`

tail -n +2 $df.csv > $df.csv.headerless
mv $df.csv.headerless $df.csv
zstd $df.csv -o $df.csv.zstd
rclone copyto -Pv $df.csv.zstd r2:rsys/database/import/$df.csv.zstd
gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json
gcloud storage cp $df.csv $bucket/

psql "$connstr" -c "DROP TABLE IF EXISTS ${db}_staging; CREATE TABLE ${db}_staging (medium INT, prefix TEXT, data BYTEA);"
gcloud sql import csv inference $bucket/$df.csv --database=postgres --table=${db}_staging -q --async
sleep 10
PENDING_OPERATION=$(gcloud sql operations list --instance=inference --filter="TYPE:IMPORT AND NOT STATUS:DONE" --format='value(name)')
gcloud sql operations wait "$PENDING_OPERATION" --timeout=unlimited
gcloud storage rm $bucket/$df.csv

psql "$connstr" -c "CREATE UNIQUE INDEX ${db}_staging_medium_prefix_idx ON ${db}_staging (medium, prefix);"
psql "$connstr" -c "BEGIN; LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE; ALTER TABLE ${db} RENAME TO ${db}_old; ALTER TABLE ${db}_staging RENAME TO ${db}; DROP TABLE ${db}_old; ALTER INDEX ${db}_staging_medium_prefix_idx RENAME TO ${db}_medium_prefix_idx; COMMIT;"
