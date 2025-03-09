#!/bin/bash
set -euxo pipefail
cd $1
bucket=`cat ../../../secrets/gcp.bucket.txt`
mlr --csv split -n 1000000 --prefix fingerprints --headerless-csv-output fingerprints.csv
gcloud auth login --quiet --cred-file=../../../secrets/gcp.auth.json
gcloud storage cp fingerprints_*.csv $bucket/
N=`ls fingerprints_*.csv | wc -l`
for i in `seq 1 $N`
do
    gcloud sql import csv inference $bucket/fingerprints_$i.csv --database=postgres --table=collect_users_staging -q
    gcloud storage rm $bucket/fingerprints_$i.csv
    rm fingerprints_$i.csv
done
