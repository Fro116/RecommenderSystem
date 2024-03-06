import HTTP
import NBInclude: @nbinclude
import Memoize: @memoize
import Oxygen
@nbinclude("../TrainingAlphas/Alpha.ipynb")
@nbinclude("../TrainingAlphas/Ensemble/EnsembleInputs.ipynb")

@memoize read_params_memoized(args...; kwargs...) = read_params(args...; kwargs...)

function compute_linear_alpha(
    metric::String,
    medium::String,
    username::String,
    source::String,
)
    alphas = get_ensemple_alphas(metric, medium)
    X = [
        get_raw_split("rec_inference", medium, [:userid], a, username, source).alpha for
        a in alphas
    ]
    if metric in ["watch", "plantowatch"]
        push!(X, fill(1.0f0 / num_items(medium), length(X[1])))
    elseif metric == "drop"
        push!(X, fill(1.0f0, length(X[1])), fill(0.0f0, length(X[1])))
    end
    X = reduce(hcat, X)
    params = read_params_memoized("$medium/Linear/$metric", true)
    e = X * params["β"]
    model(userids, itemids) = [e[x+1] for x in itemids]
    write_alpha(model, medium, "$medium/Linear/$metric", REC_SPLITS, username, source)
end

Oxygen.@get "/query" function (req::HTTP.Request)
    params = Oxygen.queryparams(req)
    username = params["username"]
    source = params["source"]
    @sync for medium in ALL_MEDIUMS
        for metric in ALL_METRICS
            Threads.@spawn compute_linear_alpha(metric, medium, username, source)
        end
    end
end

Oxygen.serveparallel(; port=parse(Int, ARGS[1]))