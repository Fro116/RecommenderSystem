import NBInclude: @nbinclude
if !@isdefined ENSEMBLE_IFNDEF
    ENSEMBLE_IFNDEF=true
    source_name = "LinearModel"
    import Logging
    import SparseArrays: sparse    
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Ensemble/EnsembleInputs.ipynb")
    using Flux
    Logging.disable_logging(Logging.Warn)
    
    function read_recommendee_suppressed_alpha(alpha::String, split::String, task::String, medium::String)
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

    function compute_linear_alpha(alpha::String, content::String, task::String, medium::String)
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

    function get_query_transform(alpha::String)
        if occursin("ItemCount", alpha) || occursin("UserVariance", alpha)
            return x -> log(x + 1)
        else
            return identity
        end
    end
    
    function get_query_features(alphas::Vector{String}, medium::String)
        A = Matrix{Float32}(undef, num_items(medium), length(alphas))
        Threads.@threads for i = 1:length(alphas)
            transform = get_query_transform(alphas[i])
            A[:, i] = transform.(read_recommendee_alpha(alphas[i], "all", medium).rating)
        end
        collect(A')
    end

    function compute_mle_alpha(name::String, medium::String)
        params = read_params(name)
        Q = get_query_features(params["hyp"].alphas, medium)
        Q = (Q .- params["inference_data"]["μ"]) ./ params["inference_data"]["σ"]
        m = params["m"]
        scores = vec(m(Q))
        write_recommendee_alpha(scores, medium, name)
    end
end
    
    
username = ARGS[1]
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_linear_alpha("$medium/$task/LinearExplicit", "explicit", task, medium)
        compute_linear_alpha("$medium/$task/LinearImplicit", "implicit", task, medium)
        for i in 0:6 # TODO
            compute_mle_alpha("$medium/$task/MLE.Ensemble.$i", medium)
        end
    end
end