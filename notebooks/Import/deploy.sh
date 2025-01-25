#!/bin/bash

kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}

trap kill_bg_processes INT
logs="../../data"
logrotate="../Collect/logrotate.jl"
mkdir -p $logs && rm $logs/*.log
(cd lists && julia save_fingerprints.jl | julia -t 1 ../$logrotate ../$logs/save_fingerprints.log) &
(cd media && julia save_media.jl| julia -t 1 ../$logrotate ../$logs/save_media.log) & 

tail -F $logs/save_fingerprints.log $logs/save_media.log
