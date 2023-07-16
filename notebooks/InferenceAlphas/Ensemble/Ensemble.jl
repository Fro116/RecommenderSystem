import NBInclude: @nbinclude
if !@isdefined ENSEMBLE_IFNDEF
    ENSEMBLE_IFNDEF = true
    source_name = "LinearModel"
    import Logging
    import SparseArrays: sparse
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Ensemble/EnsembleInputs.ipynb")
    @nbinclude("../../TrainingAlphas/Ensemble/Utility.ipynb")
    using Flux
    Logging.disable_logging(Logging.Warn)

    function read_recommendee_suppressed_alpha(
        alpha::String,
        split::String,
        task::String,
        medium::String,
    )
        @assert split == "all"
        if alpha in implicit_raw_alphas(task, medium)
            suppress = true
            content = "implicit"
        else
            suppress = false
        end
        df = read_recommendee_alpha(alpha, split, medium)
        if suppress
            seen = get_recommendee_split(content, medium)
            p_seen = sum(df.rating[seen.item])
            ϵ = sqrt(eps(Float32))
            if 1 - p_seen > ϵ
                df.rating[seen.item] .= 0
                df.rating ./= 1 - p_seen
            end
        end
        df
    end

    function compute_linear_alpha(
        alpha::String,
        content::String,
        task::String,
        medium::String,
    )
        params = read_params(alpha)
        X = []
        for alpha in params["alphas"]
            push!(X, read_recommendee_suppressed_alpha(alpha, "all", task, medium).rating)
        end
        if content == "implicit"
            push!(X, fill(1.0f0 / num_items(medium), num_items(medium)))
        end
        X = reduce(hcat, X)
        write_recommendee_alpha(X * params["β"], medium, alpha)
    end

    function get_query_features(alphas::Vector{String}, medium::String)
        A = Matrix{Float32}(undef, num_items(medium), length(alphas))
        Threads.@threads for i = 1:length(alphas)
            A[:, i] = read_recommendee_alpha(alphas[i], "all", medium).rating
        end
        collect(A')
    end

    function get_differential_query_features(ensemble_params, medium)
        dQ = []
        excludes = nothing
        for feature in ensemble_params["hyp"].alphas
            params = read_params(feature)
            seen = get_recommendee_split("implicit", medium)
            rs = []
            for i = 1:length(params["alphas"])
                alpha = params["alphas"][i]
                if occursin("ExplicitUserItemBiases", alpha) ||
                   occursin("SequelExplicit", alpha)
                    continue
                end
                rec_params = read_recommendee_params(params["alphas"][i])
                if isnothing(excludes)
                    excludes = rec_params["excludes"]
                else
                    @assert excludes == rec_params["excludes"]
                end
                r = rec_params["alpha"]
                if occursin("implicit", lowercase(alpha))
                    p_seen = sum(r[seen.item, :], dims = 1)
                    r[seen.item, :] .= 0
                    r ./= 1 .- p_seen
                end
                r = (r .- r[:, 1]) * params["β"][i]
                push!(rs, r)
            end
            rs = sum(rs)
            push!(dQ, reshape(rs, (1, size(rs)...)))
        end
        vcat(dQ...), excludes
    end

    function compute_mle_alpha(name::String, medium::String)
        params = read_params(name)
        Q = get_query_features(params["hyp"].alphas, medium)
        dQ, excludes = get_differential_query_features(params, medium)
        Q = dQ .+ Q
        Q = (Q .- params["inference_data"]["μ"]) ./ params["inference_data"]["σ"]
        m = params["m"]
        preds = dropdims(m(Q); dims = 1)
        write_recommendee_alpha(preds[:, 1], medium, name)
        write_recommendee_params(Dict("excludes" => excludes, "alpha" => preds), name)
    end
end
    
username = ARGS[1]
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_linear_alpha("$medium/$task/LinearExplicit", "explicit", task, medium)
        compute_linear_alpha("$medium/$task/LinearImplicit", "implicit", task, medium)
        for i in 0:6
            compute_mle_alpha("$medium/$task/MLE.Ensemble.$i", medium)
        end
    end
end