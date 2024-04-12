@nbinclude("notebooks/TrainingAlphas/Ensemble/EnsembleInputs.ipynb")

PARAMS = Dict(
    x => read_params(x, true) for medium in ALL_MEDIUMS for metric in ALL_METRICS for
    x in ["$medium/Linear/$metric"]
)

function compute_linear(metric::String, medium::String, alphas::Dict)
    X = [alphas[α] for α in get_ensemble_alphas(metric, medium)]
    if metric in ["watch", "plantowatch"]
        push!(X, fill(1.0f0 / num_items(medium), length(X[1])))
    elseif metric == "drop"
        push!(X, fill(1.0f0, length(X[1])), fill(0.0f0, length(X[1])))
    end
    X = reduce(hcat, X)
    params = PARAMS["$medium/Linear/$metric"]
    X * params["β"]
end


function compute_linear(payload::Dict, alphas::Dict)
    linear = Dict{String,Any}(
        "$x/Linear/$y" => nothing for x in ALL_MEDIUMS for y in ALL_METRICS
    )
    @sync for x in ALL_MEDIUMS
        for y in ALL_METRICS
            Threads.@spawn linear["$x/Linear/$y"] = compute_linear(y, x, alphas)
        end
    end
    linear
end
