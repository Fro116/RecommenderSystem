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
    if d["type"] == "user"
        return autocomplete_user(d, encoding)
    elseif d["type"] == "item"
        return autocomplete_item(d, encoding)
    else
        return HTTP.Response(400, [])
    end
end

function autocomplete_user(d::Dict, encoding)::HTTP.Response
    prefix = lowercase(sanitize(d["prefix"]))
    d = Dict(
        "source" => d["source"],
        "prefix" => prefix,
        "type" => d["type"],
    )
    r_ac = HTTP.post(
        "$DATABASE_READ_URL/read_autocomplete",
        encode(d, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_ac)
        ret = []
    else
        d_ac = decode(r_ac)
        acs = decompress(d_ac["data"])
        ret = []
        for y in acs
            x = Dict()
            x["avatar"] = y["avatar"]
            x["username"] = y["username"]
            x["matched"] = fill(false, length(y["username"]))
            x["matched"][1:length(prefix)] .= true
            x["missing_avatar"] = "https://s4.anilist.co/file/anilistcdn/user/avatar/large/default.png"
            x["age"] = nothing
            if !isnothing(y["birthday"])
                age = round(floor(time() - y["birthday"]) / 86400 / 365)
                if (age > 0 && age < 100)
                    x["age"] = age
                end
            end
            gender_map = Dict(0 => "Male", 1 => "Female", 2 => "Nonbinary")
            x["gender"] = get(gender_map, y["gender"], nothing)
            x["joined"] = nothing
            if !isnothing(y["created_at"])
                joinyear = Dates.year(Dates.unix2datetime(y["created_at"]))
                if joinyear >= 2000 && joinyear <= Dates.year(Dates.now())
                    x["joined"] = joinyear
                end
            end
            x["last_online"] = nothing
            inactivity_secs = 86400 * 365
            if !isnothing(y["last_online"])
                if time() - y["last_online"] < inactivity_secs
                    x["last_online"] = "Now"
                else
                    lastonline_year = Dates.year(Dates.unix2datetime(y["last_online"]))
                    if lastonline_year >= 2000 && lastonline_year <= Dates.year(Dates.now())
                        x["last_online"] = lastonline_year
                    end
                end
            else
                x["last_online"] = nothing
            end
            push!(ret, x)
        end
    end
    ret = Dict("prefix" => d["prefix"], "autocompletes" => ret)
    HTTP.Response(200, encode(ret, :json, encoding)...)
end

@memoize function get_autocomplete_items_map(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks = 1)
    seen = Set()
    records = []
    for i = 1:DataFrames.nrow(df)
        k = df.matchedid[i]
        if k == 0 || k ∈ seen
            continue
        end
        push!(seen, k)
        if !ismissing(df.title[i])
            r = (df.source[i], df.itemid[i], df.matchedid[i], df.title[i])
            push!(records, r)
        end
        if !ismissing(df.english_title[i])
            r = (
                df.source[i],
                df.itemid[i],
                df.matchedid[i],
                df.english_title[i],
            )
            push!(records, r)
        end
        if !ismissing(df.alternative_titles[i])
            for t in JSON3.read(df.alternative_titles[i])
                r = (df.source[i], df.itemid[i], df.matchedid[i], t)
                push!(records, r)
            end
        end
    end
    ret = Dict()
    seen_ret = Set()
    for (source, itemid, matchedid, title) in records
        seen_k = (source, itemid, lowercase(title))
        if seen_k in seen_ret
            continue
        end
        push!(seen_ret, seen_k)
        k = (source, itemid)
        if k ∉ keys(ret)
            ret[k] = []
        end
        push!(ret[k], (matchedid, title))
    end
    ret
end

function autocomplete_item(d::Dict, encoding)::HTTP.Response
    prefix = lowercase(lstrip(d["prefix"]))
    args = Dict(
        "medium" => Dict("Manga" => 0, "Anime" => 1)[d["medium"]],
        "prefix" => prefix,
        "type" => d["type"],
    )
    r_ac = HTTP.post(
        "$DATABASE_READ_URL/read_autocomplete",
        encode(args, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_ac)
        ret = []
    else
        ret = []
        d_ac = decode(r_ac)
        acs = decompress(d_ac["data"])
        titlemap = get_autocomplete_items_map(args["medium"])
        for (source, itemid, title, _) in acs
            k = (source, itemid)
            if k ∉ keys(titlemap)
                continue
            end
            for (matchedid, alt_title) in titlemap[k]
                if title != lowercase(alt_title)
                    continue
                end
                matched = fill(false, length(alt_title))
                match = findfirst(prefix, lowercase(alt_title))
                for (i, idx) in Iterators.enumerate(eachindex(alt_title))
                    if idx == match[1]
                        matched[i:i+length(prefix)-1] .= true
                    end
                end
                card = get(get_media_info(args["medium"]), matchedid, nothing)
                if isnothing(card)
                    continue
                end
                if !isnothing(card["images"])
                    image = first(card["images"])
                else
                    image = first(card["missing_images"])
                end
                entry = Dict(
                    "source" => source,
                    "itemid" => itemid,
                    "matchedid" => matchedid,
                    "matched_title" => alt_title,
                    "matched" => matched,
                    "title" => card["title"],
                    "image" => image,
                    "mediatype" => card["type"],
                    "startdate" => card["startdate"],
                    "enddate" => card["enddate"],
                    "episodes" => card["episodes"],
                    "chapters" => card["chapters"],
                )
                push!(ret, entry)
            end
        end
    end
    ret = Dict("prefix" => d["prefix"], "autocompletes" => ret)
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

@memoize function get_matchedid_map(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks = 1)
    d = Dict()
    for i = 1:DataFrames.nrow(df)
        k = (df.source[i], df.itemid[i])
        v = df.matchedid[i]
        if v == 0 || k ∈ keys(d)
            continue
        end
        d[k] = v
    end
    d
end

Oxygen.@post "/update" function update_state(r::HTTP.Request)::HTTP.Response
    encoding = nothing
    if occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        encoding = :gzip
    end
    d = decode(r)
    state = d["state"]
    action = d["action"]
    if isempty(d["state"])
        if action["type"] == "add_item"
            default_medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
        else
            default_medium = 1
        end
        state = Dict{String,Any}("medium" => default_medium, "users" => [], "items" => [])
    else
        state = MsgPack.unpack(Base64.base64decode(d["state"]))
    end
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
    elseif action["type"] == "add_item"
        source = action["source"]
        medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
        itemid = action["itemid"]
        matchedid = get(get_matchedid_map(medium), (source, itemid), nothing)
        if isnothing(matchedid)
            return HTTP.Response(404, [])
        end
        push!(state["items"], Dict("medium" => medium, "matchedid" => matchedid))
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
        return HTTP.Response(400, [])
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
    test_users = CSV.read(
        "$secretdir/test.users.csv",
        DataFrames.DataFrame,
        stringtype = String,
        ntasks = 1,
    )
    test_items = CSV.read(
        "$secretdir/test.items.csv",
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
    for (source, medium, itemid) in zip(test_items.source, test_items.medium, test_items.itemid)
        m = Dict(0=>"Manga", 1=>"Anime")[medium]
        logtag("STARTUP", "/autocomplete")
        HTTP.post(
            "http://localhost:$PORT/autocomplete",
            encode(
                Dict("medium" => m, "prefix" => source[1:1], "type" => "item"),
                :json,
                :gzip,
            )...,
            status_exception = false,
        )
        logtag("STARTUP", "/add_item")
        apply_action(Dict("type" => "add_item", "source" => source, "medium" => m, "itemid" => itemid))
    end
    for (source, username) in zip(test_users.source, test_users.username)
        logtag("STARTUP", "/autocomplete")
        HTTP.post(
            "http://localhost:$PORT/autocomplete",
            encode(
                Dict("source" => source, "prefix" => username, "type" => "user"),
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
