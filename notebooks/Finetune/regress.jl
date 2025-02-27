import Glob
import NNlib: logsoftmax, sigmoid, softmax
import Optim
import ProgressMeter: @showprogress
include("../julia_utils/http.jl")

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
    registry_data = open("../../data/finetune/model.registry.msgpack") do f
        MsgPack.unpack(read(f))
    end
    registry = Dict()
    @showprogress for d in registry_data
        registry[d["name"]] = Dict(
            "weight" => permutedims(convert.(Float32, reduce(hcat, d["weight"]))),
            "bias" => convert.(Float32, d["bias"]),
        )
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
            registry[k]["weight"] *
            convert.(Float32, reduce(hcat, [x["embeds"][k] for x in users])) .+
            registry[k]["bias"],
        ) * w for (k, w) in zip(models, weights)
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
    alphas = ["$x.$medium.$metric" for x in ["baseline", "bagofwords"]]
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
    Dict(alphas .=> coefs)
end

function save_weights()
    ret = Dict()
    for medium in [0, 1]
        for metric in ["rating", "watch", "plantowatch", "drop"]
            ret["$medium.$metric"] = regress(users, medium, metric)
        end
    end
    open("../../data/finetune/model.weights.msgpack", "w") do g
        write(g, MsgPack.pack(ret))
    end
end

const users = get_users()
const registry = get_registry()
save_weights()
