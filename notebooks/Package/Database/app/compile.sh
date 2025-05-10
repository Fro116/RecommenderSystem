#!/bin/bash
(cd /usr/src/app/layer1/notebooks/Collect && uvicorn layer1:app --host 0.0.0.0 --port 5001 --log-level warning) &
(cd /usr/src/app/layer2/notebooks/Collect && julia -t auto,auto --trace-compile trace layer2.jl 5002 10 "http://localhost:5001/proxy" true 10 false) &
(cd /usr/src/app/layer3/notebooks/Collect && julia -t auto,auto --trace-compile trace layer3.jl 5003 "http://localhost:5002" 1 1) &
(cd /usr/src/app/database/notebooks/Inference && julia -t auto,auto --trace-compile trace database.jl 5004 "http://localhost:5003" "nothing") &
sleep 30
curl http://localhost:5004/shutdown
curl http://localhost:5003/shutdown
curl http://localhost:5002/shutdown
curl http://localhost:5001/shutdown
cat /usr/src/app/*/notebooks/Collect/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Collect/trace trace
