#   Ensemble
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#     •  See the corresponding file in ../../TrainingAlphas for more
#        details

using Logging
Logging.disable_logging(Logging.Warn)
import NBInclude: @nbinclude
import SparseArrays: sparse
if !@isdefined ENSEMBLE_IFNDEF
    ENSEMBLE_IFNDEF=true
    source_name = "LinearModel"
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Ensemble/EnsembleInputs.ipynb")
    Logging.disable_logging(Logging.Warn)
    
    function read_recommendee_suppressed_alpha(alpha::String, split::String, task::String)
        @assert split == "all"
        if alpha in implicit_raw_alphas(task)
            suppress = true
            content = "implicit"
        else
            suppress = false
        end
        df = read_recommendee_alpha(alpha, split)
        if suppress
            seen = get_recommendee_split(content)
            p_seen = sum(df.rating[seen.item])
            ϵ = sqrt(eps(Float32))
            if 1 - p_seen > ϵ
                df.rating[seen.item] .= 0
                df.rating ./= 1 - p_seen
            end
        end
        df
    end

    function compute_linear_alpha(alpha::String, content::String, task::String)
        params = read_params(alpha)
        X = []
        for alpha in params["alphas"]
            push!(X, read_recommendee_suppressed_alpha(alpha, "all", task).rating)
        end
        if content == "implicit"
            push!(X, fill(1.0f0 / num_items(), num_items()))
        end
        X = reduce(hcat, X)
        write_recommendee_alpha(X * params["β"], alpha)
    end

    function get_query_transform(alpha)
        if occursin("NonlinearImplicit", alpha)
            transform = identity
        elseif occursin("ItemCount", alpha)
            transform = x -> log(x + 1)
        elseif occursin("Variance", alpha) || occursin("implicit", lowercase(alpha))
            transform = x -> log(x + Float32(eps(Float64)))
        else
            transform = identity
        end
    end
    
    function get_query_features(alphas::Vector{String})
        A = Matrix{Float32}(undef, num_items(), length(alphas))
        Threads.@threads for i = 1:length(alphas)
            transform = get_query_transform(alphas[i])
            A[:, i] = transform.(read_recommendee_alpha(alphas[i], "all").rating)
        end
        collect(A')
    end

    function compute_mle_alpha(name)
        params = read_params(name)
        Q = get_query_features(params["hyp"].alphas)
        Q = (Q .- params["inference_data"]["μ"]) ./ params["inference_data"]["σ"]
        clip_std = params["inference_data"]["clip_std"]
        clamp!(Q, -clip_std, clip_std)
        m = params["m"]
        chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i = 1:n:length(arr)]
        scores = Array{Float32}(undef, num_items())
        batch_size = params["hyp"].batch_size
        batches = chunk(convert.(Int32, 1:num_items()), batch_size)
        Threads.@threads for batch in batches
            sample = Q[:, batch]
            alpha = m(sample)
            scores[batch] .= vec(alpha)
        end
        write_recommendee_alpha(scores, name)
    end
end
    
    
username = ARGS[1]
for task in ALL_TASKS
    compute_linear_alpha("$task/LinearExplicit", "explicit", task)
    compute_linear_alpha("$task/LinearImplicit", "implicit", task)
    compute_mle_alpha("$task/MLE.Ensemble")
end
