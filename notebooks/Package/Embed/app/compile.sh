#!/bin/bash
(cd /usr/src/app/layer4/notebooks/Collect && julia -t auto,auto --trace-compile trace layer4.jl 6001 nothing) &
(cd /usr/src/app/embed_py/notebooks/Finetune && uvicorn embed:app --host 0.0.0.0 --port 6002 --log-level warning) &
(cd /usr/src/app/embed_jl/notebooks/Finetune && julia -t auto,auto --trace-compile trace embed.jl 6003 "http://localhost:6001" "http://localhost:6002") &
sleep 30
curl http://localhost:6003/shutdown
curl http://localhost:6002/shutdown
curl http://localhost:6001/shutdown
cat /usr/src/app/*/notebooks/*/trace > trace
julia /usr/src/app/compile.jl
rm /usr/src/app/*/notebooks/Finetune/trace trace
