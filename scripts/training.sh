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
export JULIA_NUM_THREADS="16"
logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
cd $workdir/RecommenderSystem/notebooks
(julia cron.jl |& julia -t 1 $logjl $logs/cron.log) &
tail -F $logs/cron.log
