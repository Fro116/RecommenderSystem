module embed

import Oxygen
import JLD2
import NNlib: logsoftmax, sigmoid
include("../Training/import_list.jl")
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

const PORT = parse(Int, ARGS[1])
const FETCH_URL = ARGS[2]
const datadir = "../../data/finetune"
const secretdir = "../../secrets"
const bluegreen = read("$datadir/bluegreen", String)
const MODEL_URL = read("$secretdir/url.embed.$bluegreen.txt", String)
const registry = JLD2.load("$datadir/model.registry.jld2")
const planned_status = 3

standardize(x::Dict) = Dict(lowercase(String(k)) => v for (k, v) in x)

@memoize function num_items(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame).matchedid) + 1
end

@memoize function get_media_info(source, medium)
    df = CSV.read("$datadir/$(source)_$(medium).csv", DataFrames.DataFrame)
    info = Dict()
    for i = 1:DataFrames.nrow(df)
        item = Dict{String, String}("title" => df.title[i], "source" => source)
        info[string(df.itemid[i])] = item
    end
    info
end

@memoize function get_media_info(medium)
    info = Dict()
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame)
    for i = 1:DataFrames.nrow(df)
        if df.matchedid[i] == 0 || df.matchedid[i] in keys(info)
            continue
        end
        try
            info[df.matchedid[i]] = get_media_info(df.source[i], m)[string(df.itemid[i])]
        catch
            logerror("get_media_info: invalid $((df.source[i], m, df.itemid[i]))")
        end
    end
    info
end

Oxygen.@post "/add_user" function add_user(r::HTTP.Request)::HTTP.Response
    d = decode(r)
    r_read = HTTP.post(
        "$FETCH_URL/read",
        encode(Dict("source" => d["source"], "username" => d["username"]), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_read)
        return HTTP.Response(r_read.status, [])
    end
    d_read = decode(r_read)
    data = decompress(d_read["data"])
    data["user"] = standardize(data["user"])
    data["items"] = standardize.(data["items"])
    u = import_user(d_read["source"], data)
    r_embed = HTTP.post(
        "$MODEL_URL/embed",
        encode(u, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_embed)
        return HTTP.Response(r_embed.status, [])
    end
    d_embed = decode(r_embed)
    d_embed["source"] = d["source"]
    d_embed["weights"] = Dict()
    for m in [0, 1]
        d_embed["$(m)_idx"] = []
        d_embed["$(m)_status"] = []
    end
    for x in u["items"]
        m = x["medium"]
        push!(d_embed["$(m)_idx"], x["matchedid"])
        push!(d_embed["$(m)_status"], x["status"])
    end
    HTTP.Response(200, encode(d_embed, :msgpack)...)
end

function compute(state)
    medium = state["medium"]
    project(user, name) = sum(
        (registry[kr]["weight"] * convert.(Float32, user[ke]) .+ registry[kr]["bias"]) *
        registry[name][kr] for ((ke, kr), w) in registry[name]
    )
    log10(x) = log(x) / log(10)
    x = zeros(Float32, num_items(medium))
    for user in state["users"]
        weight = user["weights"]
        x .+=
            project(user, "$medium.rating") *
            get(weight, "rating", 1) *
            get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.watch")) * (1 / log(10))
        get(weight, "watch", 1) * get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.plantowatch")) * (0.1 / log(10))
        get(weight, "plantowatch", 1) * get(weight, "total", 1)
        x .+= sigmoid.(project(user, "$medium.drop")) * (-10)
        get(weight, "drop", 1) * get(weight, "total", 1)
    end
    # constraints
    for user in state["users"]
        for (idx, status) in zip(user["$(medium)_idx"], user["$(medium)_status"])
            if status != planned_status
                x[idx+1] = -Inf
            end
        end
        # TODO constrain cross-related series
        # TODO constrain recaps
        # TODO constrain sequels of unwatched series
        # TODO constrain by source
    end
    info = get_media_info(medium)
    ret = []
    N = 1000
    ids = sortperm(x, rev = true) .- 1
    for i in ids
        if length(ret) == Nemb
            break
        end
        if i ∉ keys(info)
            continue
        end
        push!(ret, info[i])
    end
    ret
end

Oxygen.@post "/update" function update_state(r::HTTP.Request)::HTTP.Response
    d = decode(r)
    state = d["state"]
    action = d["action"]
    if action["type"] == "add_user"
        username = action["username"]
        source = action["source"]
        r_embed = HTTP.post(
            "http://localhost:$PORT/add_user",
            encode(Dict("source" => source, "username" => username), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r_embed)
            return HTTP.Response(r.status)
        end
        d_embed = decode(r_embed)
        if "users" ∉ keys(state)
            state["users"] = []
        end
        push!(state["users"], d_embed)
    else
        @assert false
    end
    ret = Dict("state" => state, "view" => compute(state))
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function compile(port::Integer)
    profiles = CSV.read("$secretdir/test.users.csv", DataFrames.DataFrame, stringtype = String)
    while true
        try
            r = HTTP.get("$MODEL_URL/ready")
            break
        catch
            logtag("STARTUP", "waiting for $MODEL_URL to startup")
            sleep(1)
        end
    end
    state = Dict("medium" => 0)
    for (source, username) in zip(profiles.source, profiles.username)
        action = Dict("type" => "add_user", "source" => source, "username" => username)
        r = HTTP.post(
            "http://localhost:$PORT/update",
            encode(Dict("state" => state, "action" => action), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r)
            logerror("error $(r.status)")
            continue
        end
        d = decode(r)
        state = d["state"]
        state["medium"] = 1 - state["medium"] # TODO change state
    end
end

include("../julia_utils/start_oxygen.jl")

end
