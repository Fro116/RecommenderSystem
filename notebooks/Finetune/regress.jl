import JLD2
import Glob
import HDF5
import NNlib: logsoftmax, sigmoid, softmax
import Optim
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")

function start_server(port)
    Threads.@spawn run(
        `uvicorn embed:app --host 0.0.0.0 --port $port --log-level warning`,
    )
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
    port = 3000
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
    rets
end

function get_registry()
    registry = Dict()
    HDF5.h5open("../../data/finetune/model.registry.h5", "r") do f
        for k in keys(f)
            v = read(f, k)
            if endswith(k, ".bias")
                name = k[1:end-length(".bias")]
                part = "bias"
            elseif endswith(k, ".weight")
                name = k[1:end-length(".weight")]
                part = "weight"
                v = permutedims(v)
            else
                @assert false
            end
            if name ∉ keys(registry)
                registry[name] = Dict()
            end
            registry[name][part] = v
        end
    end
    registry
end

function loss(users, models, medium, metric, model_weights)
    @assert length(models) == length(model_weights)
    planned_status = 4
    if metric == "watch"
        activation = logsoftmax
    elseif metric in ["rating", "status"]
        activation = identity
    else
        @assert false
    end
    model_preds = [
        activation(
            registry[kr]["weight"] *
            convert.(Float32, reduce(hcat, [x["embeds"][ke] for x in users])) .+
            registry[kr]["bias"],
        ) for (ke, kr) in models
    ]
    losses = zero(eltype(model_weights))
    weights = zero(eltype(model_weights))
    for i = 1:length(users)
        ys = []
        for x in users[i]["test_items"]
            if x["medium"] != medium
                continue
            end
            if metric == "watch"
                if (x["status"] == 0 || x["status"] >= planned_status) && (x["rating"] == 0 || x["rating"] >= 5)
                    push!(ys, (x["matchedid"] + 1, 1))
                end
            elseif metric in ["rating", "status"]
                if x[metric] > 0
                    push!(ys, (x["matchedid"] + 1, x[metric]))
                end
            else
                @assert false
            end
        end
        if length(ys) == 0
            continue
        end
        w = 1 / length(ys)
        for (j, y) in ys
            p = sum([mp[j, i] * mw for (mp, mw) in zip(model_preds, model_weights)])
            weights += w
            if metric == "watch"
                losses += -p * y * w
            elseif metric in ["rating", "status"]
                losses += (p - y)^2 * w
            else
                @assert false
            end
        end
    end
    losses / weights
end

function regress(users, medium, metric)
    alphas = [
        ("baseline.$medium.$metric", "baseline.$medium.$metric"),
        ("transformer.$medium", "transformer.$medium.$metric"),
    ]
    if metric == "rating"
        push!(alphas, ("bagofwords.$medium.$metric", "bagofwords.$medium.$metric"))
    end
    if metric == "watch"
        transform = softmax
    elseif metric in ["rating", "status"]
        transform = identity
    else
        @assert false
    end
    res = Optim.optimize(
        β -> loss(users, alphas, medium, metric, transform(β)),
        fill(0.0f0, length(alphas)),
        Optim.LBFGS(),
        autodiff = :forward,
        Optim.Options(
            show_trace = true,
            extended_trace = true,
            g_tol = Float64(sqrt(eps(Float32))),
            time_limit = 1800,
        ),
    )
    coefs = transform(Optim.minimizer(res))
    Dict(alphas .=> coefs), Optim.minimum(res)
end

function save_weights()
    ret = Dict()
    losses = Dict()
    for medium in [0, 1]
        for metric in ["watch", "rating", "status"]
            coefs, loss = regress(users, medium, metric)
            ret["$medium.$metric"] = coefs
            losses["$medium.$metric"] = loss
        end
    end
    ret = merge(ret, registry)
    JLD2.save("../../data/finetune/model.registry.jld2", ret)
    logtag("REGRESS", "$losses")
    open("../../data/finetune/regress.csv", "w") do f
        write(f, "task,loss\n")
        for (k, v) in losses
            write(f, "$k,$v\n")
        end
    end
end

const users = get_users()
const registry = get_registry()
save_weights()
