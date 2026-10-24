const datadir = "../../../data/training"
const tag = read("$datadir/list_tag", String)
run(`julia retrieval_dataset.jl`)
for m in [0, 1]
    run(`python retrieval_train.py --datadir ../../../data/training/search --device 0 --medium $m --features content`)
end
run(`julia metrics.jl`)
run(`rclone copyto -Pv $datadir/search/search.embeddings.jld2 r2:rsys/database/training/$tag/search.embeddings.jld2`)
