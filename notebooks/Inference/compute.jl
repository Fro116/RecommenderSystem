module embed

import Base64
import Oxygen
include("../Training/import_list.jl")
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

const PORT = parse(Int, ARGS[1])
const DATABASE_READ_URL = ARGS[2]
const DATABASE_WRITE_URL = ARGS[3]
const datadir = "../../data/finetune"
const secretdir = "../../secrets"
const bluegreen = read("$datadir/bluegreen", String)
const MODEL_URLS = (length(ARGS) >= 4) ? [ARGS[4]] : readlines("$secretdir/url.embed.$bluegreen.txt")
MODEL_URL = first(MODEL_URLS)
include("render.jl")

standardize(x::Dict) = Dict(lowercase(String(k)) => v for (k, v) in x)
sanitize(x) = strip(x)

function request(args...; kwargs...)::HTTP.Response
    try
        return HTTP.request(args...; kwargs..., connect_timeout=1)
    catch e
        if !isa(e, HTTP.ConnectError)
            return HTTP.Response(500, [])
        end
        if !update_routing_table()
            return HTTP.Response(500, [])
        end
        try
            return HTTP.request(args...; kwargs..., connect_timeout=1)
        catch
            return HTTP.Response(500, [])
        end
    end
end

Oxygen.@get "/update_routing_table" function update_routing_table(r::HTTP.Request)::HTTP.Response
    status = update_routing_table() ? 200 : 404
    HTTP.Response(status, [])
end

function update_routing_table()::Bool
    for url in MODEL_URLS
        try
            r = HTTP.get("$url/ready", status_exception=false, connect_timeout=1)
            if !HTTP.iserror(r)
                global MODEL_URL
                MODEL_URL = url
                return true
            end
        catch e
            @assert isa(e, HTTP.ConnectError)
        end
    end
    false
end

Oxygen.@get "/bluegreen" function read_bluegreen(r::HTTP.Request)::HTTP.Response
    encoding = nothing
    if occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        encoding = :gzip
    end
    HTTP.Response(200, encode(Dict("bluegreen" => bluegreen), :json, encoding)...)
end

Oxygen.@post "/autocomplete" function autocomplete(r::HTTP.Request)::HTTP.Response
    encoding = nothing
    if occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        encoding = :gzip
    end
    d = decode(r)
    r_ac = HTTP.post(
        "$DATABASE_READ_URL/read_autocomplete",
        encode(
            Dict(
                "source" => d["source"],
                "prefix" => lowercase(sanitize(d["prefix"])),
                "type" => d["type"],
            ),
            :msgpack,
        )...,
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

function add_user(d::Dict, medium::Integer)::HTTP.Response
    r_read = HTTP.post(
        "$DATABASE_READ_URL/read_user_history",
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
    u["timestamp"] = time()
    ret = Dict("user" => u, "source" => d["source"], "username" => d["username"])
    r_embed = request(
        "POST",
        "$MODEL_URL/embed?medium=$medium&task=retrieval",
        encode(u, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_embed)
        return HTTP.Response(r_embed.status, [])
    end
    d_embed = decode(r_embed)
    ret["embeds"] = d_embed
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function refresh_user(source, username)
    r_read = HTTP.post(
        "$DATABASE_WRITE_URL/refresh_user_history",
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
        state = Dict{String,Any}("medium" => 1, "users" => [])
    else
        state = MsgPack.unpack(Base64.base64decode(d["state"]))
    end
    action = d["action"]
    followup_action = nothing
    if action["type"] == "add_user"
        username = sanitize(action["username"])
        source = action["source"]
        followup_action =
            Dict("type" => "refresh_user", "source" => source, "username" => username)
        r_embed =
            add_user(Dict("source" => source, "username" => username), state["medium"])
        if r_embed.status == 404 && refresh_user(source, username)
            r_embed =
                add_user(Dict("source" => source, "username" => username), state["medium"])
            followup_action = nothing
        end
        if HTTP.iserror(r_embed)
            return HTTP.Response(r_embed.status)
        end
        d_embed = decode(r_embed)
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
        r_embed =
            add_user(Dict("source" => source, "username" => username), state["medium"])
        if HTTP.iserror(r_embed)
            return HTTP.Response(r_embed.status)
        end
        d_embed = decode(r_embed)
        state["users"][idx] = d_embed
    elseif action["type"] == "set_media"
        medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
        state["medium"] = medium
        Threads.@threads for i = 1:length(state["users"])
            u = state["users"][i]
            if "masked.$medium" in keys(u["embeds"])
                continue
            end
            r_embed = request(
                "POST",
                "$MODEL_URL/embed?medium=$medium&task=retrieval",
                encode(u["user"], :msgpack)...,
                status_exception = false,
            )
            if HTTP.iserror(r_embed)
                return HTTP.Response(r_embed.status)
            end
            d_embed = decode(r_embed)
            state["users"][i]["embeds"] = merge(state["users"][i]["embeds"], d_embed)
        end
    else
        @assert false
    end
    r, ok = render(state, d["pagination"])
    if !ok
        return r
    end
    view, total = r
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
    while true
        if !update_routing_table()
            logtag("STARTUP", "waiting for models to startup")
            sleep(10)
        else
            break
        end
    end
    r = HTTP.get("http://localhost:$PORT/bluegreen", status_exception = false)
    if HTTP.iserror(r)
        logerror("bluegreen error $(r.status)")
    end
    profiles = CSV.read(
        "$secretdir/test.users.csv",
        DataFrames.DataFrame,
        stringtype = String,
        ntasks = 1,
    )
    state = ""
    pagination = Dict("offset" => 0, "limit" => 25)
    function apply_action(action)
        r = HTTP.post(
            "http://localhost:$PORT/update",
            encode(
                Dict("state" => state, "action" => action, "pagination" => pagination),
                :json,
                :gzip,
            )...,
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
        logtag("STARTUP", "/autocomplete")
        HTTP.post(
            "http://localhost:$PORT/autocomplete",
            encode(
                Dict("source" => source, "prefix" => lowercase(username), "type" => "user"),
                :json,
                :gzip,
            )...,
            status_exception = false,
        )
        logtag("STARTUP", "/add_user")
        apply_action(Dict("type" => "add_user", "source" => source, "username" => username))
        logtag("STARTUP", "/refresh_user")
        apply_action(
            Dict("type" => "refresh_user", "source" => source, "username" => username),
        )
    end
    for m in ["Manga", "Anime"]
        apply_action(Dict("type" => "set_media", "medium" => m))
    end
end

const allowed_origins = ["Access-Control-Allow-Origin" => "*"]
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
