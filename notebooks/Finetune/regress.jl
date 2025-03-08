import JLD2
import Glob
import HDF5
import NNlib: logsoftmax, sigmoid, softmax
import Optim
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")

function start_server(port)
    t = Threads.@spawn run(
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
    users = Vector{Any}(undef, length(fns))
    Threads.@threads for i = 1:length(fns)
        data = open(fns[i]) do f
            MsgPack.unpack(read(f))
        end
        ret = HTTP.post("http://localhost:$port/embed", encode(data, :msgpack)...)
        data["embeds"] = decode(ret)
        users[i] = data
    end
    stop_server(port)
    users
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

function loss(users, models, medium, metric, weights)
    @assert length(models) == length(weights)
    planned_status = 3
    if metric == "rating"
        activation = identity
    elseif metric in ["watch", "plantowatch"]
        activation = logsoftmax
    elseif metric == "drop"
        activation = sigmoid
    else
        @assert false
    end
    preds = sum([
        activation(
            registry[kr]["weight"] *
            convert.(Float32, reduce(hcat, [x["embeds"][ke] for x in users])) .+
            registry[kr]["bias"],
        ) * w for ((ke, kr), w) in zip(models, weights)
    ])
    losses = zero(eltype(weights))
    weights = zero(eltype(weights))
    for i = 1:length(users)
        ys = []
        for x in users[i]["test_items"]
            if x["medium"] != medium
                continue
            end
            if metric == "rating"
                if x["rating"] == 0
                    continue
                end
                push!(ys, (x["matchedid"] + 1, x["rating"]))
            elseif metric == "watch"
                if x["status"] <= planned_status
                    continue
                end
                push!(ys, (x["matchedid"] + 1, 1))
            elseif metric == "plantowatch"
                if x["status"] != planned_status
                    continue
                end
                push!(ys, (x["matchedid"] + 1, 1))
            elseif metric == "drop"
                v = (x["status"] > 0 && x["status"] < planned_status) ? 1 : 0
                push!(ys, (x["matchedid"] + 1, v))
            else
                @assert false
            end
        end
        w = 1 / length(ys)
        for (j, y) in ys
            weights += w
            if metric == "rating"
                losses += (preds[j, i] - y)^2 * w
            elseif metric in ["watch", "plantowatch"]
                losses += -preds[j, i] * y * w
            elseif metric == "drop"
                losses += -(y * log(preds[j, i]) + (1 - y) * log(1 - preds[j, i])) * w
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
    if metric == "rating"
        transform = identity
    elseif metric in ["watch", "plantowatch", "drop"]
        transform = softmax
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
        for metric in ["rating", "watch", "plantowatch", "drop"]
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
