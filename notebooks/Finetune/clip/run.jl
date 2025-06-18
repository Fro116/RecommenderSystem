run(`julia clip_dataset.jl`)
run(`python clip_dataset.py`)
for m in [0, 1]
    run(`python clip.py --datadir ../../../data/finetune/clip --device 0 --medium $m`)
end
run(`julia clip.jl`)

