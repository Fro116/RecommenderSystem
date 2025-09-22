const datadir = "../../../data/training"
run(`julia pairwise_dataset.jl`)
run(`python pairwise_dataset.py`)
for m in [0, 1]
    run(`python pairwise_ltr.py --datadir ../../../data/training/item_similarity --device 0 --medium $m --features retrieval`)
    tag = read("$datadir/list_tag", String)
    for suffix in ["pt", "csv"]
        run(`rclone copyto -Pv $datadir/item_similarity/pairwise.model.$m.$suffix r2:rsys/database/training/$tag/pairwise.model.$m.$suffix`)
    end
end
