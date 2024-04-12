module App

import CodecZstd
import HTTP
import Oxygen
import MLUtils
import MsgPack
import NBInclude: @nbinclude
import SparseArrays
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")
@nbinclude("notebooks/TrainingAlphas/Baseline/BaselineHelper.ipynb")

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

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
    SparseArrays.sparse(
        df.itemid .+ 1,
        df.userid .+ 1,
        df.metric,
        num_items(medium),
        1,
    )    
end

function get_inputs(payload::Dict, alphas::Dict)
    vcat(
        [
            get_inputs(medium, metric, payload, alphas)
            for metric in ["rating", "watch"]
            for medium in ALL_MEDIUMS
        ]...
    ) 
end

function record_sparse_array!(d::Dict, name::String, x::SparseArrays.SparseMatrixCSC)
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

function wake(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

function query(req::HTTP.Request)
    payload = unpack(req.body)
    baselines = Dict{String,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        baselines[medium] = compute_baseline(payload, medium)
    end
    alphas = merge(values(baselines)...)
    features = get_features(payload, alphas)
    d = Dict{String,Any}(
        "dataset" => features,
        "alphas" => alphas,
    )
    for medium in ALL_MEDIUMS
        d["seen_$medium"] = get_raw_split(payload, medium, [:itemid], nothing).itemid
        d["ptw_$medium"] = get_split(payload, "plantowatch", medium, [:itemid], nothing).itemid
        d["num_items_$medium"] = num_items(medium)
    end
    msgpack(d)
end

function precompile_run(running::Bool, port::Int, query::String)
    if running
        return HTTP.get("http://localhost:$port$query")
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        r = HTTP.Request("GET", query, [], "")
        return fn(r)
    end
end

function precompile_run(running::Bool, port::Int, query::String, data::Vector{UInt8})
    if running
        return HTTP.post(
            "http://localhost:$port$query",
            [("Content-Type", "application/msgpack")],
            data,
        )
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        req = HTTP.Request("POST", query, [("Content-Type", "application/msgpack")], data)
        return fn(req)
    end
end

function precompile(running::Bool, port::Int)
    while true
        try
            r = precompile_run(running, port, "/wake")
            if unpack(r.body)["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end
    
    payload = pack(
        Dict(
            "anime" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[1],
                "priority" => Int32[0],
                "status" => Int32[6],
                "progress" => Float32[1.0],
                "updated_at" => Float32[1.0],
                "started_at" => Float32[0.0],
                "repeat_count" => Int32[0],
                "owned" => Int32[0],
                "sentiment" => Int32[0],
                "itemid" => Int32[0],
                "finished_at" => Float32[0.0],
                "source" => Int32[0],
                "userid" => Int32[0],
            ),
            "manga" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[0],
                "priority" => Int32[0],
                "status" => Int32[6],
                "progress" => Float32[1.0],
                "updated_at" => Float32[1.0],
                "started_at" => Float32[0.0],
                "repeat_count" => Int32[0],
                "owned" => Int32[0],
                "sentiment" => Int32[0],
                "itemid" => Int32[0],
                "finished_at" => Float32[0.0],
                "source" => Int32[0],
                "userid" => Int32[0],
            ),
        ),
    )
    precompile_run(running, port, "/query", payload)
end

end