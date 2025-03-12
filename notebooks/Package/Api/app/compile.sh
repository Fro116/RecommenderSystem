#!/bin/bash
(cd /usr/src/app/api/notebooks/Finetune && julia -t auto,auto --trace-compile trace api.jl 8080) &
sleep 30
curl http://localhost:8080/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Finetune/trace trace
