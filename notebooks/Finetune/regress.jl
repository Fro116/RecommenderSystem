import CSV
import DataFrames
import JLD2
import JSON3
import Glob
import HDF5
import LinearAlgebra
import Memoize: @memoize
import NNlib: gelu, logsoftmax, softmax
import Optim
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../Training/history_tools.jl")
include("embed.jl")

const datadir = "../../data/finetune"
const deleted_status = 3
const planned_status = 5
const port = 5000
const MODEL_URL = "http://localhost:$port"
const num_ranking_items = 1024
LinearAlgebra.BLAS.set_num_threads(1)

function get_registry()
    registry = Dict()
    HDF5.h5open("$datadir/model.registry.h5", "r") do f
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
    HTTP.get("$MODEL_URL/shutdown", status_exception = false)
end

function save_users(medium, registry)
    logtag("REGRESS", "saving users for medium $medium")
    fns = Glob.glob("$datadir/users/test/*/*.msgpack")
    users = Vector{Any}(undef, length(fns))
    m = medium
    Threads.@threads for i = 1:length(fns)
        users[i] = nothing
        fn = fns[i]
        # load user
        data = open(fn) do f
            MsgPack.unpack(read(f))
        end
        @assert length(data["test_items"]) == 1
        if only(data["test_items"])["medium"] != medium
            continue
        end
        data["timestamp"] = data["test_items"][1]["history_max_ts"]
        data["oos_items"] = data["test_items"]
        delete!(data, "test_items")
        # retrieval
        r_retrieval = query_model(
            Dict(k => data[k] for k in ["user", "items", "timestamp"]),
            medium,
            nothing,
        )
        data = merge(data, r_retrieval)
        last_status = Dict()
        for x in data["items"]
            if x["medium"] != medium
                continue
            end
            last_status[x["matchedid"]] = x["status"]
        end
        data["last_status"] = last_status
        logp = logsoftmax(
            registry["transformer.masked.$m.watch.weight"] * data["masked.$m"] +
            registry["transformer.masked.$m.watch.bias"],
        )
        # apply business rules
        logp[1] = -Inf
        for (x, s) in data["last_status"]
            if s ∉ [deleted_status, planned_status]
                logp[x+1] = -Inf
            end
        end
        idxs = sortperm(logp, rev = true)[1:num_ranking_items] .- 1
        testid = only(data["oos_items"])["matchedid"]
        if testid ∉ idxs
            pop!(idxs)
            push!(idxs, testid)
        end
        # ranking
        r_ranking = query_model(
            Dict(k => data[k] for k in ["user", "items", "timestamp"]),
            medium,
            idxs,
        )
        data = merge(data, r_ranking)
        data["ranking_matchedids"] = idxs
        users[i] = data
    end
    rets = [x for x in users if !isnothing(x)]
    records = []
    for u in rets
        project!(u)
        record = Dict()
        for k in ["masked.$m", "causal.retrieval.$m", "causal.ranking.$m"]
            record[k] = Float32.(u[k])
        end
        record["ranking_matchedids"] = u["ranking_matchedids"]
        item = only(u["oos_items"])
        record["medium"] = item["medium"]
        record["matchedid"] = item["matchedid"]
        record["rating"] = item["rating"]
        record["predict_rating"] =
            (item["rating"] > 0) && (item["rating"] != item["history_rating"])
        inferred_watch = item["status"] == 0 && isnothing(item["history_status"])
        new_watch =
            (item["status"] > planned_status) && (
                isnothing(item["history_status"]) ||
                0 < item["history_status"] <= planned_status
            )
        record["predict_watch"] = inferred_watch || new_watch
        record["status"] = item["status"]
        record["last_status"] = u["last_status"]
        record["num_tokens"] = length(u["items"])
        push!(records, record)
    end
    JLD2.save("$datadir/regress.$medium.jld2", Dict("users" => records))
end

function skip_user(u, medium::Integer, task::AbstractString)::Bool
    if task == "retrieval"
        if u["medium"] != medium || !u["predict_watch"]
            return true
        end
        if u["matchedid"] == 0
            return true
        end
        if u["matchedid"] in keys(u["last_status"]) &&
           u["last_status"][u["matchedid"]] ∉ [deleted_status, planned_status]
            return true # don't recommend items the user has already watched
        end
        return false
    elseif task == "ranking"
        if u["medium"] != medium || !u["predict_rating"]
            return true
        end
        if u["matchedid"] == 0
            return true
        end
        return false
    else
        @assert false
    end
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

function retrieval_metrics(users, registry, medium::Integer; multithreaded = true)
    if multithreaded
        chunk_size = div(length(users), Threads.nthreads(), RoundUp)
        tasks = map(Iterators.partition(users, chunk_size)) do chunk
            Threads.@spawn retrieval_metrics(chunk, registry, medium; multithreaded = false)
        end
        states = fetch.(tasks)
        ret = Dict()
        for k in keys(first(states))
            if k != "num_users"
                ret[k] =
                    sum(x[k] * x["num_users"] for x in states) /
                    sum(x["num_users"] for x in states)
            end
        end
        return ret
    end
    m = medium
    ks = [8, 128, 1024]
    hitrate = Dict(k => 0.0 for k in ks)
    ndcg = Dict(k => 0.0 for k in ks)
    num_users = 0
    for u in users
        if skip_user(u, m, "retrieval")
            continue
        end
        logp = logsoftmax(
            registry["transformer.masked.$m.watch.weight"] * u["masked.$m"] +
            registry["transformer.masked.$m.watch.bias"],
        )
        # apply business rules
        logp[1] = -Inf
        for (x, s) in u["last_status"]
            if s ∉ [deleted_status, planned_status]
                logp[x+1] = -Inf
            end
        end
        # metrics
        ys = Set(u["matchedid"] + 1)
        for k in ks
            hitrate[k] += hitrate_at_k(logp, ys, k)
            ndcg[k] += ndcg_at_k(logp, ys, k)
        end
        num_users += 1
    end
    ret = Dict()
    for k in ks
        ret["$m.retrieval.HR@$k"] = hitrate[k] / num_users
        ret["$m.retrieval.nDCG@$k"] = ndcg[k] / num_users
    end
    ret["num_users"] = num_users
    ret
end

function regress_retrieval(users, registry, medium::Integer)
    m = medium
    p_masked = zeros(Float32, length(users))
    p_causal = zeros(Float32, length(users))
    y = zeros(Float32, length(users))
    Threads.@threads for i = 1:length(users)
        u = users[i]
        if skip_user(u, m, "retrieval")
            continue
        end
        masked_logits =
            registry["transformer.masked.$m.watch.weight"] * u["masked.$m"] +
            registry["transformer.masked.$m.watch.bias"]
        causal_logits =
            registry["transformer.causal.$m.watch.weight"] * u["causal.retrieval.$m"] +
            registry["transformer.causal.$m.watch.bias"]
        p_masked[i] = softmax(masked_logits)[u["matchedid"]+1]
        p_causal[i] = softmax(causal_logits)[u["matchedid"]+1]
        y[i] = 1
    end
    p = hcat(p_masked, p_causal)
    function lossfn(x)
        c = 1 / (1 + exp(-x[1]))
        sum(-log.(max.(p * [c, 1 - c], eps(Float64))) .* y) / sum(y)
    end
    ret = Optim.optimize(lossfn, Float32[0], Optim.LBFGS())
    c = 1 / (1 + exp(-Optim.minimizer(ret)[1]))
    Dict(
        "$m.retrieval.coefs" => [c, 1 - c],
        "$m.retrieval.crossentropy" => Optim.minimum(ret),
        "$m.retrieval.num_users" => sum(y),
    )
end

function regress_ranking(users, registry, medium::Integer)
    m = medium
    x_baseline = zeros(Float32, length(users))
    x_masked = zeros(Float32, length(users))
    x_causal = zeros(Float32, length(users))
    y = zeros(Float32, length(users))
    w = zeros(Float32, length(users))
    Threads.@threads for i = 1:length(users)
        u = users[i]
        if skip_user(u, m, "ranking")
            continue
        end
        idx = u["matchedid"] + 1
        p_baseline = registry["transformer.causal.$m.rating_mean"]
        p_masked = let
            a = registry["transformer.masked.$m.watch.weight"][idx, :]
            h = vcat(u["masked.$m"], a)
            h =
                registry["transformer.masked.$m.rating.weight.1"] * h +
                registry["transformer.masked.$m.rating.bias.1"]
            h = gelu(h)
            registry["transformer.masked.$m.rating.weight.2"]' * h +
            only(registry["transformer.masked.$m.rating.bias.2"])
        end
        p_causal =
            u["causal.ranking.$m"][findfirst(==(u["matchedid"]), u["ranking_matchedids"])]
        x_baseline[i] = p_baseline
        x_masked[i] = p_masked
        x_causal[i] = p_causal
        y[i] = u["rating"]
        w[i] = 1
    end
    X = hcat(x_baseline, x_masked, x_causal)
    β = (X .* sqrt.(w)) \ (y .* sqrt.(w))
    loss = sum(w .* (X * β - y) .^ 2) / sum(w)
    Dict("$m.rating.coefs" => β, "$m.rating.mse" => loss, "$m.rating.num_users" => sum(w))
end

function get_ranking_features(users, registry, medium::Integer)
    m = medium
    users = [u for u in users if !skip_user(u, m, "retrieval")]
    p = zeros(Float32, num_ranking_items, length(users))
    r = zeros(Float32, num_ranking_items, length(users))
    y = zeros(Int32, num_ranking_items, length(users))
    w = zeros(Float32, length(users))
    Threads.@threads for i = 1:length(users)
        u = users[i]
        idxs = u["ranking_matchedids"] .+ 1
        y[findfirst(==(u["matchedid"]), u["ranking_matchedids"]), i] = 1
        w[i] = u["rating"]
        # watch feature
        masked_logits =
            registry["transformer.masked.$m.watch.weight"] * u["masked.$m"] +
            registry["transformer.masked.$m.watch.bias"]
        causal_logits =
            registry["transformer.causal.$m.watch.weight"] * u["causal.retrieval.$m"] +
            registry["transformer.causal.$m.watch.bias"]
        p_masked = softmax(masked_logits)[idxs]
        p_causal = softmax(causal_logits)[idxs]
        p[:, i] .= sum(registry["$m.retrieval.coefs"] .* [p_masked, p_causal])
        # rating feature
        r_baseline = fill(registry["transformer.causal.$m.rating_mean"], length(idxs))
        r_masked = let
            a = registry["transformer.masked.$m.watch.weight"][idxs, :]'
            u_emb = repeat(u["masked.$m"], 1, length(idxs))
            h = vcat(u_emb, a)
            h =
                registry["transformer.masked.$m.rating.weight.1"] * h .+
                registry["transformer.masked.$m.rating.bias.1"]
            h = gelu(h)
            h' * registry["transformer.masked.$m.rating.weight.2"] .+
            only(registry["transformer.masked.$m.rating.bias.2"])
        end
        r_causal = u["causal.ranking.$m"]
        r[:, i] .= sum(registry["$m.rating.coefs"] .* [r_baseline, r_masked, r_causal])
    end
    p, r, y, w
end

function weighted_ndcg(x, y, w)
    k, n = size(y)
    ndcg = 0.0
    for i = 1:n
        score = x[:, i]
        ys = Set(findall(==(1), y[:, i]))
        ndcg += ndcg_at_k(score, ys, k) * w[i]
    end
    ndcg / sum(w)
end

function ranking_metrics(users, registry, medium::Integer)
    m = medium
    p, r, y, w = get_ranking_features(users, registry, m)
    w_rating = Float32[x == 0 ? 0 : exp(1)^x for x in w]
    w_norating = Float32[1 for x in w]
    ret = Dict(
        "$m.ranking.wnDCG" => weighted_ndcg(p .* exp.(r), y, w_rating),
        "$m.ranking.nDCG" => weighted_ndcg(p .* exp.(r), y, w_norating),
        "$m.ranking.wnDCG.baseline" => weighted_ndcg(p, y, w_rating),
        "$m.ranking.nDCG.baseline" => weighted_ndcg(p, y, w_norating),
    )
end

function make_metric_dataframe(dict)
    function natural_sort_key(s::AbstractString)
        parts = []
        pattern = r"([^\d]+)|(\d+)"
        for m in eachmatch(pattern, s)
            if m.captures[1] === nothing
                push!(parts, parse(Int, m.match))
            else
                push!(parts, m.match)
            end
        end
        return parts
    end
    result = Dict{Int,Dict{Symbol,Any}}()
    for (k, v) in dict
        task, metric = match(r"^(\d+)\.(.+)$", k).captures
        task_id = parse(Int, task)
        result[task_id] = get(result, task_id, Dict{Symbol,Any}())
        result[task_id][Symbol(metric)] = v
    end
    df = DataFrames.DataFrame([merge(Dict(:medium => k), v) for (k, v) in result])
    other_cols = filter(!=(Symbol("medium")), names(df))
    sorted_cols = sort(other_cols, by = col -> natural_sort_key(String(col)))
    df = df[:, DataFrames.Cols(:medium, sorted_cols...)]
    for col in names(df)
        if all(x -> x isa AbstractVector, df[!, col])
            df[!, col] = JSON3.write.(df[!, col])
        end
    end
    df
end

function save_weights()
    registry = get_registry()
    metrics = Dict()
    start_server(port)
    for medium in [0, 1]
        save_users(medium, registry)
    end
    stop_server(port)
    for medium in [0, 1]
        users = JLD2.load("$datadir/regress.$medium.jld2")["users"]
        registry = merge(
            registry,
            retrieval_metrics(users, registry, medium),
            regress_retrieval(users, registry, medium),
            regress_ranking(users, registry, medium),
        )
        registry = merge(registry, ranking_metrics(users, registry, medium))
    end
    df =
        make_metric_dataframe(Dict(k => v for (k, v) in registry if first(k) in ['0', '1']))
    CSV.write("../../data/finetune/regress.csv", df)
    JLD2.save("../../data/finetune/model.registry.jld2", merge(registry, metrics))
end

save_weights()