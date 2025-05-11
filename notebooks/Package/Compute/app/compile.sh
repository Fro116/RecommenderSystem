#!/bin/bash
(cd /usr/src/app/database/notebooks/Inference && julia -t auto,auto --trace-compile trace database.jl 6001 nothing) &
(cd /usr/src/app/compute/notebooks/Inference && julia -t auto,auto --trace-compile trace compute.jl 6002 "http://localhost:6001" "https://database-769423729111.us-central1.run.app") &
sleep 30
curl http://localhost:6002/shutdown
curl http://localhost:6001/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Inference/trace trace
