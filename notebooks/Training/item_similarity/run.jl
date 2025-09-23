const datadir = "../../../data/training"
const tag = read("$datadir/list_tag", String)
run(`julia pairwise_dataset.jl`)
run(`python pairwise_dataset.py`)
for m in [0, 1]
    run(`python pairwise_ltr.py --datadir ../../../data/training/item_similarity --device 0 --medium $m --features retrieval`)
end
run(`julia pairwise_metrics.jl`)
run(`rclone copyto -Pv $datadir/pairwise.embeddings.jld2 r2:rsys/database/training/$tag/pairwise.embeddings.jld2`)
run(`rclone copyto -Pv $datadir/pairwise.embeddings.csv r2:rsys/database/training/$tag/pairwise.embeddings.csv`)
