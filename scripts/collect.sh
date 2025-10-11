#!/bin/bash
kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}
trap kill_bg_processes INT
cd ../../
workdir=`pwd`
logs="$workdir/RecommenderSystem/logs/collect"
mkdir -p $logs && rm -f $logs/*.log
source venv/bin/activate
export JULIA_PROJECT="$workdir/juliaenv"
export JULIA_NUM_THREADS="15,1"
logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
ulimit -S -n 4096
cd RecommenderSystem/notebooks/Collect
./deploy.sh
