#!/bin/sh
(cd /usr/src/app/compute/notebooks/Inference && julia -t auto,auto --trace-compile trace compute.jl) &
sleep 30
curl http://localhost:8080/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Inference/trace trace
