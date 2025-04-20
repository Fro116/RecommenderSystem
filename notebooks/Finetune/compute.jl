module embed

import Base64
import Oxygen
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
include("render.jl")

standardize(x::Dict) = Dict(lowercase(String(k)) => v for (k, v) in x)
sanitize(x) = strip(x)

Oxygen.@post "/autocomplete" function autocomplete(r::HTTP.Request)::HTTP.Response
    encoding = nothing
    if occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        encoding = :gzip
    end
    d = decode(r)
    r_ac = HTTP.post(
        "$READ_URL/autocomplete",
        encode(Dict("source" => d["source"], "prefix" => lowercase(sanitize(d["prefix"])), "type" => d["type"]), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_ac)
        acs = []
    else
        d_ac = decode(r_ac)
        acs = decompress(d_ac["data"])
        for x in acs
            x["matched"] = fill(false, length(x["username"]))
            x["matched"][1:length(d["prefix"])] .= true
            x["missing_avatar"] = "https://s4.anilist.co/file/anilistcdn/user/avatar/large/default.png"
        end
    end
    ret = Dict("prefix" => d["prefix"], "autocompletes" => acs)
    HTTP.Response(200, encode(ret, :json, encoding)...)
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
    u = import_user(d_read["source"], data, d_read["db_refreshed_at"])
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
        username = sanitize(action["username"])
        source = action["source"]
        followup_action = Dict("type" => "refresh_user", "source" => source, "username" => username)
        r_embed = HTTP.post(
            "http://localhost:$PORT/add_user",
            encode(Dict("source" => source, "username" => username), :msgpack)...,
            status_exception = false,
        )
        if r_embed.status == 404 && refresh_user(source, username)
            r_embed = HTTP.post(
                "http://localhost:$PORT/add_user",
                encode(Dict("source" => source, "username" => username), :msgpack)...,
                status_exception = false,
            )
            followup_action = nothing
        end
        if HTTP.iserror(r_embed)
            return HTTP.Response(r_embed.status)
        end
        d_embed = decode(r_embed)
        if "users" ∉ keys(state)
            state["users"] = []
        end
        push!(state["users"], d_embed)
    elseif action["type"] == "refresh_user"
        username = sanitize(action["username"])
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
    view, total = render(state, d["pagination"])
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
    pagination = Dict("offset" => 0,  "limit" => 25)
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
        HTTP.post(
            "http://localhost:$PORT/autocomplete",
            encode(Dict("source" => source, "prefix" => lowercase(username), "type" => "user"), :json, :gzip)...,
            status_exception = false,
        )
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
    "Access-Control-Allow-Methods" => "GET, POST",
    "Access-Control-Max-Age" => 600,
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
