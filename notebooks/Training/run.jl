include("../julia_utils/stdout.jl")

function prune_directories(db::String, keep::Int)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    tags = sort([chop(x) for x in split(str) if endswith(x, "/")])
    while length(tags) > keep
        todelete = popfirst!(tags)
        run(`rclone --retries=10 purge r2:rsys/database/$db/$todelete`)
    end
end

function check_gpu_success()
    list_tag = read("../../data/training/list_tag", String)
    finished_file = read(
        `rclone --retries=10 ls r2:rsys/database/training/$list_tag/transformer.causal.finished`,
        String,
    )
    success = !isempty(finished_file)
    if success
        for fn in ["transformer.$modeltype.$stem" for modeltype in ["causal", "masked"] for stem in ["csv", "pt"]]
            run(`rclone --retries=10 copyto r2:rsys/database/training/$list_tag/$fn ../../data/training/$fn`)
        end
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
    run(`rclone --retries=10 copyto ../../data/training/list_tag r2:rsys/database/training/latest`)
    prune_directories("training", 2)
end

train(ARGS[1])
