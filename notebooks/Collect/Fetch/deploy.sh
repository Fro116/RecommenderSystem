#!/bin/bash

kill_bg_processes() {
  trap '' INT TERM
  kill -INT 0
  wait
}

trap kill_bg_processes INT
logs="../../../data"
mkdir -p $logs && rm $logs/*
(uvicorn layer1:app --host 0.0.0.0 --port 4001 --log-level warning |& julia logrotate.jl $logs/layer1.log) &
(julia -t auto layer2.jl 4002 1 "http://localhost:4001/proxy" true 10 "../../../environment" |& julia logrotate.jl $logs/layer2.log) &
(julia -t auto layer3.jl 4003 "http://localhost:4002" 1000 |& julia logrotate.jl $logs/layer3.log) &
(julia -t auto collect_single.jl mal_userids userid "http://localhost:4003/mal_username" 10 |& julia logrotate.jl $logs/mal_userids.log) &
(julia -t auto collect_single.jl animeplanet_userids userid "http://localhost:4003/animeplanet_username" 10 |& julia logrotate.jl $logs/animeplanet_userids.log) &
tail -F $logs/layer1.log $logs/layer2.log $logs/layer3.log $logs/mal_userids.log $logs/animeplanet_userids.log
