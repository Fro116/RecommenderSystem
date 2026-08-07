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
export JULIA_NUM_THREADS="auto"

logjl="$workdir/RecommenderSystem/notebooks/Collect/logrotate.jl"
name=$1
logs="$workdir/RecommenderSystem/logs/$name"
mkdir -p $logs && rm -f $logs/*.log
cd $workdir/RecommenderSystem/scripts
(julia $name.jl |& julia -t 1 $logjl $logs/$name.log) &
tail -F $logs/$name.log
