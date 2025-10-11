#!/bin/bash
kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}
trap kill_bg_processes INT
name=$1
cd ../../
workdir=`pwd`
logs="$workdir/RecommenderSystem/logs/$name"
mkdir -p $logs && rm -f $logs/*.log
source venv/bin/activate
export JULIA_PROJECT="$workdir/juliaenv"
export JULIA_NUM_THREADS="16"
logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
cd $workdir/RecommenderSystem/scripts
(julia $name.jl |& julia -t 1 $logjl $logs/$name.log 1000000) &
tail -F $logs/inference.log
