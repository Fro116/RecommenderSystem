#!/usr/bin/env bash
set -uo pipefail

GROUP="server-autoscale-group"
REGION="$(tr -d '[:space:]' < secrets/aws.region.txt)"
POLL_INTERVAL=30
RETRY_DELAY=300
LOCKFILE="/tmp/refresh_asg.${GROUP}.lock"
PREFS='{"MinHealthyPercentage": 100, "InstanceWarmup": 600}'
export AWS_SHARED_CREDENTIALS_FILE="$PWD/secrets/aws.credentials"
export AWS_PAGER=""

log() { echo "[$(date '+%F %T')] $*"; }

exec 9>"$LOCKFILE"
if ! flock -n 9; then
    log "refresh for $GROUP is already running; dropping this call"
    exit 0
fi

active_refresh_id() {
    aws autoscaling describe-instance-refreshes \
        --auto-scaling-group-name "$GROUP" --region "$REGION" --max-records 1 \
        --query 'InstanceRefreshes[?contains(`["Pending","InProgress","Baking","Cancelling","RollbackInProgress"]`, Status)] | [0].InstanceRefreshId' \
        --output text 2>/dev/null
}

attempt=0
while true; do
    attempt=$((attempt + 1))

    if id=$(aws autoscaling start-instance-refresh \
        --auto-scaling-group-name "$GROUP" --region "$REGION" \
        --preferences "$PREFS" \
        --query InstanceRefreshId --output text); then
        log "attempt $attempt: started refresh $id"
    else
        id=$(active_refresh_id)
        if [ -z "$id" ] || [ "$id" = "None" ]; then
            log "attempt $attempt: could not start refresh; retrying in ${RETRY_DELAY}s"
            sleep "$RETRY_DELAY"
            continue
        fi
        log "attempt $attempt: a refresh is already active ($id); following it"
    fi

    while true; do
        sleep "$POLL_INTERVAL"
        if ! out=$(aws autoscaling describe-instance-refreshes \
            --auto-scaling-group-name "$GROUP" --region "$REGION" \
            --instance-refresh-ids "$id" \
            --query 'InstanceRefreshes[0].[Status,PercentageComplete,StatusReason]' \
            --output text); then
            log "could not read status of $id; checking again"
            continue
        fi
        IFS=$'\t' read -r status pct reason <<<"$out"
        log "refresh $id: $status (${pct}%) ${reason}"
        case "$status" in
            Successful)
                log "instance refresh succeeded"
                exit 0
                ;;
            Failed | Cancelled | RollbackSuccessful | RollbackFailed)
                break
                ;;
        esac
    done

    log "refresh $id did not succeed; retrying in ${RETRY_DELAY}s"
    sleep "$RETRY_DELAY"
done
