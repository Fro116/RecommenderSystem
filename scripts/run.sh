#!/bin/bash
kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}
trap kill_bg_processes INT

cd ../../
workdir=`pwd`
source venv/bin/activate
export JULIA_PROJECT="$workdir/juliaenv"
export JULIA_NUM_THREADS="16"
if [ "$name" = "training" ]; then
    export JULIA_NUM_THREADS="32"
fi;

logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
name=$1
logs="$workdir/RecommenderSystem/logs/$name"
mkdir -p $logs && rm -f $logs/*.log
if [ "$name" = "database" ]; then
    export project=`cat $workdir/RecommenderSystem/secrets/gcp.project.txt`
    export region=`cat $workdir/RecommenderSystem/secrets/gcp.region.txt`
    export auth="$workdir/RecommenderSystem/secrets/gcp.auth.json"
    (cloud-sql-proxy $project:$region:inference -p 6543 --credentials-file $auth |& julia -t 1 $logjl $logs/cloudsql.log) &
fi
cd $workdir/RecommenderSystem/scripts
(julia $name.jl |& julia -t 1 $logjl $logs/$name.log 1000000) &
tail -F $logs/$name.log
