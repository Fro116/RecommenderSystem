#!/bin/bash
kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}
trap kill_bg_processes INT
cd ../../
workdir=`pwd`
logs="$workdir/RecommenderSystem/logs/training"
mkdir -p $logs && rm -f $logs/*.log
source venv/bin/activate
export JULIA_PROJECT="$workdir/juliaenv"
export JULIA_NUM_THREADS="15,1"
export project=`cat $workdir/RecommenderSystem/secrets/gcp.project.txt`
export region=`cat $workdir/RecommenderSystem/secrets/gcp.region.txt`
export auth="$workdir/RecommenderSystem/secrets/gcp.auth.json"
logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
(cloud-sql-proxy $project:$region:inference -p 6543 --credentials-file $auth |& julia -t 1 $logjl $logs/cloudsql.log) &
sudo systemctl start postgresql
cd RecommenderSystem/notebooks/Collect
./deploy.sh
