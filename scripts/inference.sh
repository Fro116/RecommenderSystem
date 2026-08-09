#!/bin/bash
logs="../logs/inference"
rotate="tee ../$logs"
mkdir -p "$logs" && rm -f "$logs"/*.log
trap 'trap "" INT TERM; sleep 3; kill -KILL 0' INT TERM

(cd ../notebooks/Finetune && uvicorn embed:app --host 0.0.0.0 --port 6001 --log-level debug --timeout-graceful-shutdown 3 |& $rotate/embed.log) &
(cd ../notebooks/Collect && uvicorn layer1:app --host 0.0.0.0 --port 6002 --log-level warning --timeout-graceful-shutdown 3 |& $rotate/layer1.log) &
(cd ../notebooks/Collect && julia -t 3,1 layer2.jl 6003 10 "http://localhost:6002/proxy" true 10 false |& $rotate/layer2.log) &
(cd ../notebooks/Collect && julia -t 3,1 layer3.jl 6004 "http://localhost:6003" 1 3 |& $rotate/layer3.log) &
(cd ../notebooks/Inference && julia -t 3,1 database.jl 6005 "http://localhost:6004" |& $rotate/database.log) &
(cd ../notebooks/Inference && julia -t auto,auto compute.jl 6006 "http://localhost:6001" "http://localhost:6005" false |& $rotate/compute.log) &
(cd ../notebooks/Package/Client/app && npm run dev -- --host |& tee "../../$logs/client.log") &

tail -F $logs/embed.log $logs/layer1.log $logs/layer2.log $logs/layer3.log $logs/database.log $logs/compute.log $logs/client.log
