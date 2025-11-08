include("../julia_utils/stdout.jl")

import CSV
import DataFrames

const datadir = "../../data/training"

function prune_directories(db::String, keep::Int)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    tags = sort([chop(x) for x in split(str) if endswith(x, "/")])
    while length(tags) > keep
        todelete = popfirst!(tags)
        run(`rclone --retries=10 purge r2:rsys/database/$db/$todelete`)
    end
end

function upload_metrics()
    name = "training.usermodel.csv"
    training_tag = read("$datadir/list_tag", String)
    ts = time()
    dfs = []
    for modeltype in ["masked", "causal"]
        df = CSV.read("$datadir/transformer.$modeltype.csv")
        cols = DataFrames.names(df)
        df[!, "training_tag"] .= training_tag
        df[!, "modeltype"] .= modeltype
        df[!, "updated_at"] .= ts
        df = df[:, [["training_tag", "modeltype"]; cols; ["updated_at"]]]
        push!(dfs, df)
    end
    df = reduce(DataFrames.vcat, dfs)
    CSV.write("$datadir/$name", df)

    run(`rclone --retries=10 copyto r2:rsys/database/import/metrics.$name $datadir/metrics.$name`)
    df = CSV.read("$datadir/$name", DataFrames.DataFrame)
    if ispath("$datadir/metrics.$name")
        historical_df = CSV.read("$datadir/metrics.$name", DataFrames.DataFrame)
        training_tag = only(Set(df[:, :training_tag]))
        filter!(x -> x[:training_tag] != training_tag, historical_df)
        df = DataFrames.vcat(historical_df, df)
        sort!(df, by=x->x[:training_tag])
    end
    CSV.write("$datadir/metrics.$name", df)
    run(`rclone --retries=10 copyto $datadir/metrics.$name r2:rsys/database/import/metrics.$name`)
end

function check_gpu_success()
    list_tag = read("$datadir/list_tag", String)
    finished_file = read(
        `rclone --retries=10 ls r2:rsys/database/training/$list_tag/transformer.causal.finished`,
        String,
    )
    success = !isempty(finished_file)
    if success
        for fn in ["transformer.$modeltype.$stem" for modeltype in ["causal", "masked"] for stem in ["csv", "pt"]]
            run(`rclone --retries=10 copyto r2:rsys/database/training/$list_tag/$fn $datadir/$fn`)
        end
        upload_metrics()
    end
    success
end

function train(datetag::AbstractString)
    logtag("TRAIN", "running on $datetag")
    run(`julia import_data.jl $datetag`)
    for m in [0, 1]
        run(`julia media_relations.jl $m`)
    end
    run(`julia transformer.jl`)
    run(`julia rungpu.jl`)
    if !check_gpu_success()
        logerror("rungpu failed")
        exit(1)
    end
    cmd = "cd item_similarity && julia run.jl")
    run(`sh -c $cmd`)
    run(`rclone --retries=10 copyto $datadir/list_tag r2:rsys/database/training/latest`)
    prune_directories("training", 2)
end

train(ARGS[1])
