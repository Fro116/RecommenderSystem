#!/bin/bash
# update database tables
set -euxo pipefail
secretdir=$1

# Probe each connection string; use the first one that answers.
connstr=""
while IFS= read -r candidate; do
  [ -z "$candidate" ] && continue
  if PGCONNECT_TIMEOUT=5 psql "$candidate" -c "SELECT 1" >/dev/null 2>&1; then
    connstr="$candidate"
    break
  fi
done < "$secretdir/db.inference.txt"

if [ -z "$connstr" ]; then
  echo "ERROR: no connection string in db.inference.txt is reachable from this host" >&2
  exit 1
fi

refresh_table() {
  local fn="$1" db="$2" idx_ddl="$3" idx_staging="$4" idx_final="$5"

  psql "$connstr" -v ON_ERROR_STOP=1 \
    -c "DROP TABLE IF EXISTS ${db}_staging;" \
    -c "CREATE UNLOGGED TABLE ${db}_staging (LIKE $db);"

  rclone cat r2:rsys/database/import/$fn.zstd \
    | unzstd -c \
    | psql "$connstr" -v ON_ERROR_STOP=1 \
        -c "\copy ${db}_staging FROM STDIN WITH (FORMAT csv, HEADER true)"

  psql "$connstr" -v ON_ERROR_STOP=1 <<EOF
SET maintenance_work_mem = '1GB';
$idx_ddl
BEGIN;
LOCK TABLE ${db} IN ACCESS EXCLUSIVE MODE;
ALTER TABLE ${db} RENAME TO ${db}_old;
ALTER TABLE ${db}_staging RENAME TO ${db};
DROP TABLE ${db}_old;
ALTER INDEX ${idx_staging} RENAME TO ${idx_final};
COMMIT;
EOF
}

refresh_table "item_autocomplete.csv" "autocomplete_items" \
  "CREATE UNIQUE INDEX autocomplete_items_staging_medium_prefix_idx ON autocomplete_items_staging (medium, prefix);" \
  "autocomplete_items_staging_medium_prefix_idx" \
  "autocomplete_items_medium_prefix_idx"

refresh_table "user_autocomplete.csv" "autocomplete_users" \
  "CREATE UNIQUE INDEX autocomplete_users_staging_source_prefix_idx ON autocomplete_users_staging (source, prefix);" \
  "autocomplete_users_staging_source_prefix_idx" \
  "autocomplete_users_source_prefix_idx"

refresh_table "user_histories.csv" "user_histories" \
  "CREATE INDEX user_histories_staging_source_lower_username_idx ON user_histories_staging (source, lower(username));" \
  "user_histories_staging_source_lower_username_idx" \
  "user_histories_source_lower_username_idx"

psql "$connstr" -v ON_ERROR_STOP=1 \
  -c "DELETE FROM online_user_histories WHERE db_refreshed_at < extract(epoch from NOW()) - 86400 * 30;"