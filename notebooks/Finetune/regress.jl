import CSV
import DataFrames
import JLD2
import JSON3
import Glob
import HDF5
import LinearAlgebra
import NNlib: logsoftmax, gelu
import ProgressMeter: @showprogress
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")

function start_server(port)
    Threads.@spawn run(`uvicorn embed:app --host 0.0.0.0 --port $port --log-level warning`)
    while true
        try
            HTTP.get("http://localhost:$port/ready")
            break
        catch
            sleep(1)
        end
    end
end

function stop_server(port)
    HTTP.get("http://localhost:$port/shutdown", status_exception = false)
end

function get_users()
    port = 5000
    start_server(port)
    fns = Glob.glob("../../data/finetune/users/test/*/*.msgpack")
    nchunks = 16
    chunks = Iterators.partition(fns, div(length(fns), nchunks))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            users = []
            for fn in chunk
                data = open(fn) do f
                    MsgPack.unpack(read(f))
                end
                ret =
                    HTTP.post("http://localhost:$port/embed", encode(data, :msgpack)...)
                data["embeds"] = decode(ret)
                push!(users, data)
            end
            users
        end
    end
    rets = reduce(vcat, fetch.(tasks))
    stop_server(port)
    for u in rets
        for (k, v) in u["embeds"]
            if typeof(v) == Vector{Any}
                u["embeds"][k] = Float32.(v)
            end
        end
    end
    rets
end

function get_registry()
    registry = Dict()
    HDF5.h5open("../../data/finetune/model.registry.h5", "r") do f
        for k in keys(f)
            v = read(f, k)
            v = convert.(Float32, v)
            if length(size(v)) == 2
                if size(v)[2] == 1
                    v = vec(v)
                else
                    v = collect(transpose(v))
                end
            end
            registry[k] = convert.(Float32, v)
        end
    end
    registry
end

function hitrate_at_k(p::Vector, seen_items::Set, k::Int)
    topk = partialsortperm(p, rev = true, 1:k)
    any(i -> i in seen_items, topk) ? 1.0 : 0.0
end

function ndcg_at_k(p::Vector, seen_items::Set, k::Int)
    topk = partialsortperm(p, rev = true, 1:k)
    dcg = 0.0
    for (rank, idx) in enumerate(topk)
        if idx in seen_items
            dcg += 1 / log2(rank + 1)
        end
    end
    max_hits = min(k, length(seen_items))
    idcg = sum(1 / log2(i + 1) for i = 1:max_hits)
    idcg == 0 ? 0.0 : dcg / idcg
end

function crossentropy_loss(p::Vector, seen_items::Set)
    losses = 0
    for y in seen_items
        losses += -p[y]
    end
    losses / length(seen_items)
end

function retrieval_metrics(users, registry, medium::Integer, k::Integer)
    deleted_status = 3
    planned_status = 5
    m = medium
    crossentropy = 0.0
    crossentropy_users = 0
    hitrate = 0.0
    ndcg = 0.0
    num_users = 0
    @showprogress for u in users
        # get targets
        test_items = Set()
        for x in u["test_items"]
            if x["medium"] != m
                continue
            end
            inferred_watch = x["status"] == 0 && isnothing(x["history_status"])
            new_watch =
                (x["status"] > planned_status) && (
                    isnothing(x["history_status"]) ||
                    0 < x["history_status"] <= planned_status
                )
            if inferred_watch || new_watch
                push!(test_items, x["matchedid"] + 1)
            end
        end
        if length(test_items) == 0
            continue
        end
        # recreate transformer.py loss function
        logp = logsoftmax(
            registry["transformer.$m.embedding"] * u["embeds"]["transformer.$m"] +
            registry["transformer.$m.watch.bias"],
        )
        crossentropy += crossentropy_loss(logp, test_items)
        crossentropy_users += 1

        # apply business rules
        last_status = Dict()
        for x in u["items"]
            if x["medium"] != m
                continue
            end
            last_status[x["matchedid"]+1] = x["status"]
        end
        ys = Set()
        for x in test_items
            if x == 1
                # default id for longtail items
                continue
            end
            if x in keys(last_status) && last_status[x] ∉ [deleted_status, planned_status]
                # don't recommend items the user has already watched
                continue
            end
            push!(ys, x)
        end
        if length(ys) == 0
            continue
        end
        logp[1] = -Inf # default id for longtail items
        for (x, s) in last_status
            # don't recommend items the user has already watched
            if s ∉ [deleted_status, planned_status]
                logp[x] = -Inf
            end
        end

        # metrics
        hitrate += hitrate_at_k(logp, ys, k)
        ndcg += ndcg_at_k(logp, ys, k)
        num_users += 1
    end
    Dict(
        "$m.retrieval.HR@$k" => hitrate / num_users,
        "$m.retrieval.nDCG@$k" => ndcg / num_users,
        "$m.retrieval.crossentropy" => crossentropy / crossentropy_users,
        "$m.retrieval.num_users" => num_users,
    )
end

function rating_metrics(users, registry, medium)
    m = medium
    x_baseline = Float32[]
    x_bagofwords = Float32[] # TODO ablate bagofwords
    x_transformer_baseline = Float32[]
    x_transformer = Float32[]
    y = Float32[]
    w = Float32[]
    num_users = 0
    @showprogress for i = 1:length(users)
        u = users[i]
        count = 0
        for x in u["test_items"]
            if x["medium"] != m
                continue
            end
            predict_rating = (x["rating"] > 0) && (x["rating"] != x["history_rating"])
            if !predict_rating
                continue
            end
            idx = x["matchedid"] + 1
            p_baseline =
                only(registry["baseline.$m.rating.weight"]) *
                only(u["embeds"]["baseline.$m.rating"]) +
                registry["baseline.$m.rating.bias"][idx]
            p_bagofwords =
                registry["bagofwords.$m.rating.weight"][idx, :]' *
                u["embeds"]["bagofwords.$m.rating"] +
                registry["bagofwords.$m.rating.bias"][idx]
            p_transformer = let
                a = registry["transformer.$m.embedding"][idx, :]
                h = vcat(u["embeds"]["transformer.$m"], a)
                h =
                    registry["transformer.$m.rating.weight.1"] * h +
                    registry["transformer.$m.rating.bias.1"]
                h = gelu(h)
                registry["transformer.$m.rating.weight.2"]' * h +
                only(registry["transformer.$m.rating.bias.2"])
            end
            p_transformer_baseline = registry["baseline.$m.rating.bias"][idx]
            push!(x_baseline, p_baseline)
            push!(x_bagofwords, p_bagofwords)
            push!(x_transformer_baseline, p_transformer_baseline)
            push!(x_transformer, p_transformer)
            push!(y, x["rating"])
            count += 1
        end
        for _ = 1:count
            push!(w, 1 / count)
        end
        if count > 0
            num_users += 1
        end
    end
    X = reduce(hcat, [x_baseline, x_bagofwords, x_transformer_baseline, x_transformer])
    β = (X .* sqrt.(w)) \ (y .* sqrt.(w))
    loss = sum((X * β - y) .^ 2 .* w) / sum(w)
    Dict(
        "$m.rating.coefs" => β,
        "$m.rating.mse" => loss,
        "$m.rating.num_users" => num_users,
    )
end

function make_metric_dataframe(dict)
    result = Dict{Int,Dict{Symbol,Any}}()
    for (k, v) in dict
        task, metric = match(r"^(\d+)\.(.+)$", k).captures
        task_id = parse(Int, task)
        result[task_id] = get(result, task_id, Dict{Symbol,Any}())
        result[task_id][Symbol(metric)] = v
    end
    df = DataFrames.DataFrame([merge(Dict(:medium => k), v) for (k, v) in result])
    sorted_cols = sort(filter(!=(Symbol("medium")), names(df)))
    df = df[:, DataFrames.Cols(:medium, sorted_cols...)]
    for col in names(df)
        if all(x -> x isa AbstractVector, df[!, col])
            df[!, col] = JSON3.write.(df[!, col])
        end
    end
    df
end

function save_weights()
    users = get_users()
    registry = get_registry()
    metrics = Dict()
    for m in [0, 1]
        metrics = merge(metrics, retrieval_metrics(users, registry, m, 10))
        metrics = merge(metrics, rating_metrics(users, registry, m))
    end
    df = make_metric_dataframe(metrics)
    CSV.write("../../data/finetune/regress.csv", df)
    JLD2.save("../../data/finetune/model.registry.jld2", merge(registry, metrics))
end

save_weights()
