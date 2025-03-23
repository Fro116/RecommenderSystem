module embed

import Base64
import Oxygen
import JLD2
import NNlib: logsoftmax, sigmoid
import SparseArrays
include("../Training/import_list.jl")
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

const PORT = parse(Int, ARGS[1])
const READ_URL = ARGS[2]
const FETCH_URL = ARGS[3]
const datadir = "../../data/finetune"
const secretdir = "../../secrets"
const bluegreen = read("$datadir/bluegreen", String)
const MODEL_URL = read("$secretdir/url.embed.$bluegreen.txt", String)
const registry = JLD2.load("$datadir/model.registry.jld2")
const relations = merge([JLD2.load("$datadir/media_relations.$m.jld2") for m in [0, 1]]...)
const planned_status = 3
const watching_status = 5

standardize(x::Dict) = Dict(lowercase(String(k)) => v for (k, v) in x)

@memoize function num_items(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

function get_url(source, medium, itemid)
    medium_map = Dict(0 => "manga", 1 => "anime")
    source_map = Dict(
        "mal" => "https://myanimelist.net",
        "anilist" => "https://anilist.co",
        "kitsu" => "https://kitsu.app",
        "animeplanet" => "https://anime-planet.com",
    )
    join([source_map[source], medium_map[medium], itemid], "/")
end

@memoize function get_media_info(medium)
    info = Dict()
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks=1)
    optint(x) = x != 0 ? x : missing
    function jsonlist(x)
        if ismissing(x) || isempty(x)
            return missing
        end
        join(JSON3.read(x), ", ")
    end
    function duration(x)
        if ismissing(x) || round(x) == 0
            return missing
        end
        d = []
        if x > 60
            hours = Int(div(x, 60))
            push!(d, "$hours hr.")
            x -= hours * 60
        end
        if x >= 1
            minutes = Int(floor(x))
            push!(d, "$minutes min.")
            x -= minutes
        end
        if isempty(d)
            seconds = Int(round(x * 60))
            push!(d, "$seconds sec.")
        end
        join(d, " ")
    end
    function season(x)
        if ismissing(x)
            return missing
        end
        season_str, year = split(x, "-")
        uppercasefirst(season_str) * " " * year
    end
    for i = 1:DataFrames.nrow(df)
        if df.matchedid[i] == 0 || df.matchedid[i] in keys(info)
            continue
        end
        info[df.matchedid[i]] = Dict(
            "title" => df.title[i],
            "url" => get_url(df.source[i], medium, df.itemid[i]),
            "type" => df.mediatype[i],
            "startdate" => df.startdate[i],
            "enddate" => df.enddate[i],
            "episodes" => optint(df.episodes[i]),
            "duration" => duration(df.duration[i]),
            "chapters" => optint(df.chapters[i]),
            "volumes" => optint(df.volumes[i]),
            "status" => df.status[i],
            "season" => season.(df.season[i]),
            "studios" => jsonlist(df.studios[i]),
            "source" => df.source_material[i],
            "genres" => jsonlist(df.genres[i]),
            "synopsis" => df.synopsis[i],
        )
    end
    info
end

Oxygen.@post "/add_user" function add_user(r::HTTP.Request)::HTTP.Response
    d = decode(r)
    r_read = HTTP.post(
        "$READ_URL/read",
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
    d_embed["username"] = d["username"]
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

function compute(state, pagination)
    medium = state["medium"]
    project(user, name) = sum(
        (registry[kr]["weight"] * convert.(Float32, user[ke]) .+ registry[kr]["bias"]) *
        w for ((ke, kr), w) in registry[name]
    )
    x = zeros(Float32, num_items(medium))
    for user in state["users"]
        weight = user["weights"]
        x .+=
            project(user, "$medium.rating") *
            get(weight, "rating", 1) *
            get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.watch")) * (1 / log(4)) * get(weight, "watch", 1) * get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.plantowatch")) * (0 / log(4)) * get(weight, "plantowatch", 1) * get(weight, "total", 1)
        x .+= sigmoid.(project(user, "$medium.drop")) * (0) * get(weight, "drop", 1) * get(weight, "total", 1)
    end
    # constraints
    function get_watched(user, m, status_filter = x -> x != planned_status)
        v = zeros(Float32, num_items(m))
        for (idx, status) in zip(user["$(m)_idx"], user["$(m)_status"])
            if status_filter(status)
                v[idx+1] = 1
            end
        end
        v
    end
    for user in state["users"]
        w = get_watched(user, medium)
        # seen
        for i in 1:length(x)
            if w[i] > 0
                x[i] = -Inf
            end
        end
        # adaptations
        v1 = relations["$(medium).adaptations"] * get_watched(user, 1 - medium)
        v2 = relations["$(medium).dependencies"] * w
        for i in 1:length(x)
            if v1[i] != 0 && v2[i] == 0
                x[i] = -Inf
            end
        end
        # recaps
        v = relations["$(medium).recaps"] * w
        for i in 1:length(x)
            if v[i] != 0
                x[i] = -Inf
            end
        end
        # dependencies
        v1 = relations["$(medium).dependencies"] * w
        v2 = relations["$(medium).dependencies"] * ones(Float32, num_items(medium))
        for i in 1:length(x)
            if v2[i] != 0 && v1[i] == 0
                x[i] = -Inf
            end
        end
        # sequels to currently watching / dropped
        v = relations["$(medium).dependencies"] * get_watched(user, medium, x -> x == watching_status || x > 0 && x < planned_status)
        for i in 1:length(x)
            if v[i] != 0
                x[i] = -Inf
            end
        end
        # TODO constrain by source
    end
    info = get_media_info(medium)
    ids = sortperm(x, rev = true) .- 1
    ids = [i for i in ids if i in keys(info) && x[i+1] > -Inf]
    total = length(ids)
    sidx = pagination["offset"] + 1
    eidx = pagination["offset"] + pagination["limit"]
    if sidx > total
        view = []
    else
        view = [info[i] for i in ids[sidx:min(eidx, total)]]
    end
    view, total
end

function refresh_user(source, username)
    r_read = HTTP.post(
        "$FETCH_URL/refresh",
        encode(Dict("source" => source, "username" => username), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_read)
        return false
    end
    d = decode(r_read)
    d["refresh"]
end

Oxygen.@post "/update" function update_state(r::HTTP.Request)::HTTP.Response
    encoding = nothing
    if occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        encoding = :gzip
    end
    d = decode(r)
    state = d["state"]
    if isempty(d["state"])
        state = Dict{String, Any}("medium" => 1)
    else
        state = MsgPack.unpack(Base64.base64decode(d["state"]))
    end
    action = d["action"]
    followup_action = nothing
    if action["type"] == "add_user"
        username = action["username"]
        source = action["source"]
        r_embed = HTTP.post(
            "http://localhost:$PORT/add_user",
            encode(Dict("source" => source, "username" => username), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r_embed)
            return HTTP.Response(r_embed.status)
        end
        d_embed = decode(r_embed)
        if "users" ∉ keys(state)
            state["users"] = []
        end
        push!(state["users"], d_embed)
        followup_action = Dict("type" => "refresh_user", "source" => source, "username" => username)
    elseif action["type"] == "refresh_user"
        username = action["username"]
        source = action["source"]
        idx = nothing
        for (i, x) in Iterators.enumerate(state["users"])
            if x["username"] == username && x["source"] == source
                idx = i
                break
            end
        end
        if isnothing(idx) || !refresh_user(source, username)
            return HTTP.Response(200, encode(Dict(), :json, encoding)...)
        end
        r_embed = HTTP.post(
            "http://localhost:$PORT/add_user",
            encode(Dict("source" => source, "username" => username), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r_embed)
            return HTTP.Response(r_embed.status)
        end
        d_embed = decode(r_embed)
        state["users"][idx] = d_embed
    elseif action["type"] == "set_media"
        medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
        state["medium"] = medium
    else
        @assert false
    end
    view, total = compute(state, d["pagination"])
    ret = Dict(
        "state" => Base64.base64encode(MsgPack.pack(state)),
        "view" => view,
        "total" => total,
        "medium" => Dict(0 => "Manga", 1 => "Anime")[state["medium"]],
        "followup_action" => followup_action,
    )
    HTTP.Response(200, encode(ret, :json, encoding)...)
end

function compile(port::Integer)
    profiles = CSV.read("$secretdir/test.users.csv", DataFrames.DataFrame, stringtype = String, ntasks=1)
    while true
        try
            r = HTTP.get("$MODEL_URL/ready")
            break
        catch
            logtag("STARTUP", "waiting for $MODEL_URL to startup")
            sleep(1)
        end
    end
    state = ""
    pagination = Dict("offset" => 0,  "limit" => 50)
    function apply_action(action)
        r = HTTP.post(
            "http://localhost:$PORT/update",
            encode(Dict("state" => state, "action" => action, "pagination" => pagination), :json, :gzip)...,
            status_exception = false,
            decompress = false,
        )
        if HTTP.iserror(r)
            logerror("error $(r.status)")
            return
        end
        d = decode(r)
        if isempty(d)
            return
        end
        state = d["state"]
    end
    for (source, username) in zip(profiles.source, profiles.username)
        apply_action(Dict("type" => "add_user", "source" => source, "username" => username))
        apply_action(Dict("type" => "refresh_user", "source" => source, "username" => username))
    end
    for m in ["Manga", "Anime"]
        apply_action(Dict("type" => "set_media", "medium" => m))
    end
end

const allowed_origins = [ "Access-Control-Allow-Origin" => "*" ]
const cors_headers = [
    allowed_origins...,
    "Access-Control-Allow-Headers" => "*",
    "Access-Control-Allow-Methods" => "GET, POST"
]
function CorsHandler(handle)
    return function (req::HTTP.Request)
        if HTTP.method(req) == "OPTIONS"
            return HTTP.Response(200, cors_headers)
        else
            r = handle(req)
            append!(r.headers, allowed_origins)
            return r
        end
    end
end
const MIDDLEWARE = [CorsHandler]

include("../julia_utils/start_oxygen.jl")

end
