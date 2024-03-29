module App

import HTTP
import JSON
import Oxygen
import MLUtils
import NBInclude: @nbinclude
import SparseArrays: AbstractSparseArray, sparse
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")
@nbinclude("notebooks/TrainingAlphas/Baseline/BaselineHelper.ipynb")

# Baseline

num_threads() = 1
get_training_counts(df, col) = get_counts(getfield(df, col))

function get_item_weights(medium, df, λ, counts)
    w = zeros(Float32, length(df.itemid))
    for i = 1:length(df.itemid)
        a = df.itemid[i]
        if a ∈ keys(counts)
            w[i] = counts[a]
        end
    end
    powerdecay(w, log(λ))
end

function get_weights(medium, df, λ_wu, λ_wa, λ_wt, item_counts)
    user_weight = powerdecay(get_training_counts(df, :userid), log(λ_wu))
    item_weight = get_item_weights(medium, df, λ_wa, item_counts)
    timestamp_weight = λ_wt .^ (1 .- df.updated_at)
    user_weight .* item_weight .* timestamp_weight
end

function train_model(medium, training, λ, μ, a, item_counts)
    if length(training.rating) == 0
        return μ
    end
    λ_u, _, λ_wu, λ_wa, λ_wt = λ
    users, items, ratings = training.userid, training.itemid, training.rating
    weights = get_weights(medium, training, λ_wu, λ_wa, λ_wt, item_counts)
    u = zeros(eltype(λ_u), 1)
    ρ_u = zeros(eltype(u), length(u), Threads.nthreads())
    Ω_u = zeros(eltype(u), length(u), Threads.nthreads())
    update_users!(users, items, ratings, weights, u, a, λ_u, ρ_u, Ω_u; μ = μ)
    u
end

PARAMS = Dict(x => read_params(x, false) for x in ["$m/Baseline/rating" for m in ALL_MEDIUMS])

function compute_baseline(payload::Dict, medium::String)
    training = get_split(
        payload,
        "rating",
        medium,
        [:userid, :itemid, :rating, :update_order, :updated_at],
        nothing,
    )
    params = PARAMS["$medium/Baseline/rating"]
    u = train_model(
        medium,
        training,
        params["λ"],
        mean(params["u"]),
        params["a"],
        params["item_counts"],
    )
    p = make_prediction(
        fill(0, num_items(medium)),
        collect(0:num_items(medium)-1),
        u,
        params["a"],
    )
    Dict("$medium/Baseline/rating" => p)
end

# BagOfWords

function get_inputs(medium::String, metric::String, payload::Dict, alphas::Dict)
    split = "rec_training"
    fields = [:userid, :itemid, :metric]
    if metric == "rating"
        α = "$medium/Baseline/rating"
        β = PARAMS[α]["β"]
        df = get_split(payload, metric, medium, fields)
        df.metric .= df.metric - alphas[α][df.itemid.+=1] .* β
    else
        df = get_split(payload, metric, medium, fields)
    end
    sparse(df, medium)
end

function get_inputs(payload::Dict, alphas::Dict)
    inputs = [
        get_inputs(medium, metric, payload, alphas) for metric in ["rating", "watch"]
        for medium in ALL_MEDIUMS
    ]
    vcat(inputs...)
end

function record_sparse_array!(d::Dict, name::String, x::AbstractSparseArray)
    i, j, v = SparseArrays.findnz(x)
    d[name*"_i"] = i
    d[name*"_j"] = j
    d[name*"_v"] = v
    d[name*"_size"] = [size(x)[1], size(x)[2]]
end

function get_features(payload::Dict, alphas::Dict)
    d = Dict{String,Any}()
    X = get_inputs(payload, alphas)
    record_sparse_array!(d, "inputs", X)
    d
end

function compute_bagofwords(payload::Dict, embeddings::Dict, medium::String)
    seen = get_raw_split(payload, medium, [:itemid], nothing).itemid
    ptw = get_split(payload, "plantowatch", medium, [:itemid], nothing).itemid
    alphas = Dict{String,Any}()
    for metric in ALL_METRICS
        e = convert.(Float32, embeddings["$(medium)_$(metric)"])
        if metric in ["watch", "plantowatch"]
            r = copy(e)
            r[seen.+1] .= 0
            r = r ./ sum(r)
            p = copy(e)
            watched = setdiff(Set(seen), Set(ptw))
            p[watched.+1] .= 0
            p = p ./ sum(p)
            e = copy(r)
            e[ptw.+1] .= p[ptw.+1]
            e = copy(r)
            e[ptw.+1] .= p[ptw.+1]
        elseif metric in ["rating", "drop"]
            nothing
        else
            @assert false
        end
        alphas["$(medium)/BagOfWords/v1/$metric"] = copy(e)
    end
    alphas
end

# Oxygen

function wake(req::HTTP.Request)
    Oxygen.json(Dict("success" => true))
end

function process(req::HTTP.Request)
    payload = JSON.parse(String(req.body))
    baselines = Dict{String,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        baselines[medium] = compute_baseline(payload, medium)
    end
    alphas = merge(values(baselines)...)
    features = get_features(payload, alphas)
    Oxygen.json(Dict("alpha" => alphas, "features" => features))
end

function compute(req::HTTP.Request)
    d = JSON.parse(String(req.body))
    payload = d["payload"]
    embeddings = d["embeddings"]
    baselines = Dict{String,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        baselines[medium] = compute_bagofwords(payload, embeddings, medium)
    end
    Oxygen.json(merge(values(baselines)...))
end

function precompile(port::Int)
    while true
        try
            r = HTTP.get("http://localhost:$port/wake")
            json = JSON.parse(String(copy(r.body)))
            if json["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end

    payload = (
        "{\"anime\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0]," *
        "\"update_order\":[0],\"sentiment_score\":[0],\"medium\":[1],\"backward_order\":[1]," *
        "\"priority\":[0],\"progress\":[1.0],\"forward_order\":[1],\"status\":[6]," *
        "\"updated_at\":[1.0],\"started_at\":[0.0],\"repeat_count\":[0],\"owned\":[0]," *
        "\"sentiment\":[0],\"finished_at\":[0.0],\"source\":[0],\"unit\":[1],\"userid\":[0]}," *
        "\"manga\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0],\"update_order\":[0]," *
        "\"sentiment_score\":[0],\"medium\":[0],\"backward_order\":[1],\"priority\":[0]," *
        "\"progress\":[1.0],\"forward_order\":[1],\"status\":[6],\"updated_at\":[1.0]," *
        "\"started_at\":[0],\"repeat_count\":[0],\"owned\":[0],\"sentiment\":[0]," *
        "\"finished_at\":[0],\"source\":[0],\"unit\":[1],\"userid\":[0]}}"
    )
    HTTP.post(
        "http://localhost:$port/process",
        [("Content-Type", "application/json")],
        payload,
    )

    embeddings = Dict()
    for medium in ALL_MEDIUMS
        for metric in ALL_METRICS
            embeddings["$(medium)_$(metric)"] = ones(Float32, num_items(medium))
        end
    end
    d = Dict()
    d["payload"] = JSON.parse(payload)
    d["embeddings"] = embeddings
    HTTP.post(
        "http://localhost:$port/compute",
        [("Content-Type", "application/json")],
        JSON.json(d),
    )
end

end