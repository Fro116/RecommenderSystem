run(`julia clip_dataset.jl full`)
run(`python clip_dataset.py`)
for m in [0, 1]
    run(`python ltr.py --datadir ../../../data/finetune/clip --device 0 --medium $m --features all`)
end
run(`julia clip.jl`)

