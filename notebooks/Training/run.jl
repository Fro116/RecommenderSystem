include("../julia_utils/stdout.jl")

function prune_directories(db::String, keep::Int)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    tags = sort([chop(x) for x in split(str) if endswith(x, "/")])
    while length(tags) > keep
        todelete = popfirst!(tags)
        run(`rclone --retries=10 purge r2:rsys/database/$db/$todelete`)
    end
end


function train(datetag::AbstractString)
    logtag("TRAIN", "running on $datetag")
    run(`julia import_data.jl $datetag`)
    for m in [0, 1]
        run(`julia media_relations.jl $m`)
    end
    for mini in [true, false]
        run(`julia transformer.jl`)
    end
    run(`julia rungpu.jl`)
    cmd = "cd item_similarity && julia run.jl")
    run(`sh -c $cmd`)
    run(`rclone --retries=10 copyto ../../data/training/list_tag r2:rsys/database/training/latest`)
    prune_directories("training", 2)
end

train(ARGS[1])
