#!/bin/bash
(cd /usr/src/app/layer4/notebooks/Collect && julia -t auto,auto --trace-compile trace layer4.jl 6001 nothing) &
(cd /usr/src/app/compute/notebooks/Finetune && julia -t auto,auto --trace-compile trace compute.jl 6002 "http://localhost:6001" "https://fetch-2ppiozhuba-uc.a.run.app") &
sleep 30
curl http://localhost:6002/shutdown
curl http://localhost:6001/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Finetune/trace trace
