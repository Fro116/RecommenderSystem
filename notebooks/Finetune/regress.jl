import JLD2
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
    nchunks = 64
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
            if k == "version"
                continue
            end
            u["embeds"][k] = Vector{Float32}(v)
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

function watch_loss(users, registry, medium)
    planned_status = 4
    m = medium
    losses = 0.0
    weights = 0.0
    @showprogress for i = 1:length(users)
        u = users[i]
        ys = []
        for x in u["test_items"]
            if x["medium"] != m
                continue
            end
            if (x["status"] == 0 || x["status"] >= planned_status) &&
               (x["rating"] == 0 || x["rating"] >= 5)
                push!(ys, x["matchedid"] + 1)
            end
        end
        if length(ys) == 0
            continue
        end
        preds = logsoftmax(
            registry["transformer.$m.embedding"] * u["embeds"]["transformer.$m"] +
            registry["transformer.$m.watch.bias"],
        )
        w = 1 / length(ys)
        for y in ys
            losses += -preds[y] * w
            weights += w
        end
    end
    losses / weights
end

function rating_regress(users, registry, medium)
    x_baseline = Float32[]
    x_bagofwords = Float32[]
    x_transformer = Float32[]
    y = Float32[]
    w = Float32[]
    m = medium
    @showprogress for i = 1:length(users)
        u = users[i]
        count = 0
        for x in u["test_items"]
            if x["medium"] != m
                continue
            end
            if x["rating"] == 0
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
            push!(x_baseline, p_baseline)
            push!(x_bagofwords, p_bagofwords)
            push!(x_transformer, p_transformer)
            push!(y, x["rating"])
            count += 1
        end
        for _ = 1:count
            push!(w, 1 / count)
        end
    end
    X = reduce(hcat, [x_baseline, x_bagofwords, x_transformer])
    β = (X .* sqrt.(w)) \ (y .* sqrt.(w))
    loss = sum((X * β - y) .^ 2 .* w) / sum(w)
    β, loss
end

function save_weights()
    users = get_users()
    registry = get_registry()
    ret = Dict()
    for m in [0, 1]
        ret["$m.watch.loss"] = watch_loss(users, registry, m)
        β, loss = rating_regress(users, registry, m)
        ret["$m.rating.loss"] = loss
        ret["$m.rating.coefs"] = β
    end
    ret = merge(ret, registry)
    losses = Dict(
        "$medium.$metric" => ret["$medium.$metric.loss"] for medium in [0, 1] for
        metric in ["watch", "rating"]
    )
    JLD2.save("../../data/finetune/model.registry.jld2", ret)
    logtag("REGRESS", "$losses")
    open("../../data/finetune/regress.csv", "w") do f
        write(f, "task,loss\n")
        for (k, v) in losses
            write(f, "$k,$v\n")
        end
    end
end

save_weights()
