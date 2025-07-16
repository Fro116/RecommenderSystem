#!/bin/sh
(cd /usr/src/app/compute/notebooks/Inference && julia -t auto,auto --trace-compile trace compute.jl 6001 ) &
sleep 30
curl http://localhost:6001/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Inference/trace trace
