include("../julia_utils/stdout.jl")

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
end

train(ARGS[1])
