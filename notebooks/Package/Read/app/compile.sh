#!/bin/bash
(cd /usr/src/app/layer4/notebooks/Collect && julia -t auto,auto --trace-compile trace layer4.jl 5004 nothing "../../secrets/test.users.csv") &
sleep 30
curl http://localhost:5004/shutdown
cat /usr/src/app/*/notebooks/Collect/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Collect/trace trace
