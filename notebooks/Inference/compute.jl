module embed

import Base64
import Oxygen
import UUIDs
import Random
import Serialization
include("../Training/import_list.jl")
include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
include("../Finetune/embed.jl")

const secretdir = "../../secrets"
const PORT = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8080
const DATABASE_WRITE_URL = length(ARGS) >= 2 ? ARGS[2] : read("$secretdir/url.database.txt", String)
const datadir = "../../data/finetune"
const MODEL_URL = (length(ARGS) >= 3) ? ARGS[3] : read("$secretdir/url.embed.txt", String)
include("render.jl")

sanitize(x) = strip(x)

const serverid = UUIDs.uuid4()

function difftime(speedscope)
    tags = [x[1] for x in speedscope[2:end]]
    times = diff([x[2] for x in speedscope])
    total = speedscope[end][2] - speedscope[1][2]
    [total, serverid, collect(zip(tags, times))]
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

function read_autocomplete(data::Dict)::HTTP.Response
    if data["type"] == "user"
        return read_autocomplete_user(data)
    elseif data["type"] == "item"
        return read_autocomplete_item(data)
    end
    HTTP.Response(400, [])
end

function read_autocomplete_user(data::Dict)::HTTP.Response
    source = data["source"]
    prefix = data["prefix"]
    table = "autocomplete_users"
    df = with_db(:inference_read, 3) do db
        query = "SELECT * FROM $table WHERE (source, prefix) = (\$1, \$2)"
        stmt = db_prepare(db, query)
        DataFrames.DataFrame(LibPQ.execute(stmt, (source, prefix)))
    end
    if df isa Symbol || DataFrames.nrow(df) == 0
        return HTTP.Response(404, [])
    end
    d = Dict(k => only(df[:, k]) for k in DataFrames.names(df))
    HTTP.Response(200, encode(d, :msgpack)...)
end

function read_autocomplete_item(data::Dict)::HTTP.Response
    medium = data["medium"]
    prefix = data["prefix"]
    table = "autocomplete_items"
    df = with_db(:inference_read, 3) do db
        query = "SELECT * FROM $table WHERE (medium, prefix) = (\$1, \$2)"
        stmt = db_prepare(db, query)
        DataFrames.DataFrame(LibPQ.execute(stmt, (medium, prefix)))
    end
    if df isa Symbol || DataFrames.nrow(df) == 0
        return HTTP.Response(404, [])
    end
    d = Dict(k => only(df[:, k]) for k in DataFrames.names(df))
    HTTP.Response(200, encode(d, :msgpack)...)
end

function autocomplete_user(d::Dict, encoding)::HTTP.Response
    speedscope = [("start", time())]
    prefix = lowercase(sanitize(d["prefix"]))
    d = Dict(
        "source" => d["source"],
        "prefix" => prefix,
        "type" => d["type"],
    )
    r_ac = read_autocomplete(d)
    push!(speedscope, ("read", time()))
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
    push!(speedscope, ("finish", time()))
    ret = Dict("prefix" => d["prefix"], "autocompletes" => ret, "speedscope" => difftime(speedscope))
    HTTP.Response(200, encode(ret, :json, encoding)...)
end

@memoize function get_autocomplete_items_map(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks = 1)
    sort!(df, :count, rev=true)
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

function temporally_consistent_sample(arr::Vector, salt::String, window::Real)
    t = time()
    seed = hash((floor(Int, t / window), salt))
    rand(Random.Xoshiro(seed), arr)
end

function autocomplete_item(d::Dict, encoding)::HTTP.Response
    speedscope = [("start", time())]
    prefix = lowercase(lstrip(d["prefix"]))
    args = Dict(
        "medium" => Dict("Manga" => 0, "Anime" => 1)[d["medium"]],
        "prefix" => prefix,
        "type" => d["type"],
    )
    r_ac = read_autocomplete(args)
    push!(speedscope, ("read", time()))
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
                salt = string((args["medium"], k))
                if !isnothing(card["images"])
                    image = temporally_consistent_sample(card["images"], salt, 60)
                else
                    image = temporally_consistent_sample(card["missing_images"], salt, 60)
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
    push!(speedscope, ("finish", time()))
    ret = Dict("prefix" => d["prefix"], "autocompletes" => ret, "speedscope" => difftime(speedscope))
    HTTP.Response(200, encode(ret, :json, encoding)...)
end

function get_user_url(source, username, userid)
    if source == "mal"
        return "https://myanimelist.net/profile/$username"
    elseif source == "anilist"
        return "https://anilist.co/user/$userid"
    elseif source == "kitsu"
        return "https://kitsu.app/users/$userid"
    elseif source == "animeplanet"
        return "https://anime-planet.com/users/$username"
    else
        @assert false
    end
end

function read_user_history(data::Dict)
    source = data["source"]
    username = data["username"]
    tables = get(data, "tables", ["user_histories", "online_user_histories"])
    allow_online = get(data, "online_history", false)
    tasks = []
    for table in tables
        task = Threads.@spawn begin
            df = with_db(:inference_read, 3) do db
                query = "SELECT * FROM $table WHERE (source, lower(username)) = (\$1, lower(\$2))"
                stmt = db_prepare(db, query)
                DataFrames.DataFrame(LibPQ.execute(stmt, (source, username)))
            end
        end
        push!(tasks, task)
    end
    dfs = []
    for task in tasks
        df = fetch(task)
        if df isa Symbol || DataFrames.nrow(df) == 0
            continue
        end
        push!(dfs, df)
    end
    if isempty(dfs)
        return HTTP.Response(404, []), false
    end
    df = reduce(vcat, dfs)
    sort!(df, :db_refreshed_at, rev=true)
    d = Dict(k => df[1, k] for k in DataFrames.names(df))
    d, true
end

function add_user(d::Dict, medium::Int, speedscope)
    d_read, ok = read_user_history(Dict("source" => d["source"], "username" => d["username"]))
    if !ok
        return d_read, false
    end
    push!(speedscope, ("dbread", time()))
    data = decompress(d_read["data"])
    push!(speedscope, ("decompress", time()))
    u = import_user(d_read["source"], data, d_read["db_refreshed_at"])
    u["timestamp"] = time()
    usermap = data["usermap"]
    ret = Dict(
        "user" => u,
        "source" => d["source"],
        "username" => d["username"],
        "header" => Dict(
            "titlename" => usermap["username"],
            "titleurl" => get_user_url(d["source"], usermap["username"], usermap["userid"]),
        ),
    )
    push!(speedscope, ("import", time()))
    d_embed = query_model(u, medium, nothing)
    if isnothing(d_embed)
        return HTTP.Response(500, []), false
    end
    push!(speedscope, ("retrieval_model", time()))
    ret["embeds"] = d_embed
    ret, true
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

function decode_state(r::HTTP.Request)
    speedscope = [("start", time())]
    d = decode(r)
    state = d["state"]
    if isempty(d["state"])
        state = Dict(
            "medium" => 1,
            "users" => [],
            "items" => [],
            "penalties" => Dict(
                "same_series_penalty" => 0,
                "related_penalty" => 0,
                "mmr_penalty" => 0,
            )
        )
        uri = HTTP.URI(r.target)
        if uri.path == "/add_user"
            state["penalties"] = Dict(
                "same_series_penalty" => 1,
                "related_penalty" => 1,
                "mmr_penalty" => 1,
            )
        elseif uri.path == "/add_item"
            state["penalties"] = Dict(
                "same_series_penalty" => 0,
                "related_penalty" => 0,
                "mmr_penalty" => 0,
            )
        else
            logerror("decode_state: unknown path $(uri.path)")
        end
    else
        state = Serialization.deserialize(IOBuffer(Base64.base64decode(d["state"])))
    end
    pagination = d["pagination"]
    encoding = get_preferred_encoding(r)
    action = d["action"]
    push!(speedscope, ("state", time()))
    state, action, pagination, encoding, speedscope
end

function render_state(state, pagination, encoding, followup_action, speedscope)
    push!(speedscope, ("prerender", time()))
    r, ok = render(state, pagination, speedscope)
    if !ok
        return r
    end
    push!(speedscope, ("render", time()))
    view, total = r
    state_str = let io = IOBuffer(); Serialization.serialize(io, state); Base64.base64encode(take!(io)) end
    ret = Dict(
        "state" => state_str,
        "view" => view,
        "total" => total,
        "medium" => Dict(0 => "Manga", 1 => "Anime")[state["medium"]],
        "followup_action" => followup_action,
    )
    header = first(vcat(state["users"], state["items"]))["header"]
    ret["titlename"] = header["titlename"]
    ret["titleurl"] = header["titleurl"]
    push!(speedscope, ("finish", time()))
    ret["speedscope"] = difftime(speedscope)
    HTTP.Response(200, encode(ret, :json, encoding)...)
end

Oxygen.@post "/add_user" function add_user_endpoint(r::HTTP.Request)::HTTP.Response
    state, action, pagination, encoding, speedscope = decode_state(r)
    username = sanitize(action["username"])
    source = action["source"]
    followup_action =
        Dict("endpoint" => "/refresh_user", "source" => source, "username" => username)
    r_embed, ok =
        add_user(Dict("source" => source, "username" => username), state["medium"], speedscope)
    if !ok && r_embed.status == 404 && refresh_user(source, username)
        r_embed, ok =
            add_user(Dict("source" => source, "username" => username), state["medium"], speedscope)
        followup_action = nothing
    end
    if !ok
        return HTTP.Response(r_embed.status)
    end
    push!(state["users"], r_embed)
    render_state(state, pagination, encoding, followup_action, speedscope)
end

Oxygen.@post "/refresh_user" function refresh_user_endpoint(r::HTTP.Request)::HTTP.Response
    state, action, pagination, encoding, speedscope = decode_state(r)
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
    r_embed, ok =
        add_user(Dict("source" => source, "username" => username), state["medium"], speedscope)
    if !ok
        return HTTP.Response(r_embed.status)
    end
    state["users"][idx] = r_embed
    render_state(state, pagination, encoding, nothing, speedscope)
end

Oxygen.@post "/add_item" function add_item_endpoint(r::HTTP.Request)::HTTP.Response
    state, action, pagination, encoding, speedscope = decode_state(r)
    source = action["source"]
    medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
    state["medium"] = medium
    itemid = action["itemid"]
    matchedid = get(get_matchedid_map(medium), (source, itemid), nothing)
    if isnothing(matchedid)
        return HTTP.Response(404, [])
    end
    card = get(get_media_info(medium), matchedid, nothing)
    if isnothing(card)
        return HTTP.Response(404, [])
    end
    d_item = Dict(
        "medium" => medium,
        "matchedid" => matchedid,
        "header" => Dict(
            "titlename" => card["title"],
            "titleurl" => card["url"],
        ),
    )
    push!(state["items"], d_item)
    render_state(state, pagination, encoding, nothing, speedscope)
end

Oxygen.@post "/set_media" function set_media_endpoint(r::HTTP.Request)::HTTP.Response
    state, action, pagination, encoding, speedscope = decode_state(r)
    medium = Dict("Manga" => 0, "Anime" => 1)[action["medium"]]
    state["medium"] = medium
    Threads.@threads for i = 1:length(state["users"])
        u = state["users"][i]
        if "masked.$medium" in keys(u["embeds"])
            continue
        end
        d_embed = query_model(u["user"], medium, nothing)
        if isnothing(d_embed)
            return HTTP.Response(500, [])
        end
        state["users"][i]["embeds"] = merge(state["users"][i]["embeds"], d_embed)
    end
    render_state(state, pagination, encoding, nothing, speedscope)
end


function compile_source(port::Integer, compile_source::AbstractString)
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
    pagination = Dict("offset" => 0, "limit" => 16)
    function apply_action(endpoint, action)
        headers, data = encode(
            Dict("state" => state, "action" => action, "pagination" => pagination),
            :json,
            :gzip,
        )
        headers["Accept-Encoding"] = rand(["gzip", "zstd"])
        r = HTTP.post(
            "http://localhost:$PORT/$endpoint",
            headers,
            data,
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
        if source != compile_source
            continue
        end
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
        logtag("STARTUP", "/add_item $compile_source")
        apply_action("add_item", Dict("source" => source, "medium" => m, "itemid" => itemid))
    end
    for (source, username) in zip(test_users.source, test_users.username)
        if source != compile_source
            continue
        end
        logtag("STARTUP", "/autocomplete $compile_source")
        HTTP.post(
            "http://localhost:$PORT/autocomplete",
            encode(
                Dict("source" => source, "prefix" => username, "type" => "user"),
                :json,
                :gzip,
            )...,
            status_exception = false,
        )
        logtag("STARTUP", "/add_user $compile_source")
        apply_action("add_user", Dict("source" => source, "username" => username))
        logtag("STARTUP", "/refresh_user $compile_source")
        apply_action("refresh_user", Dict("source" => source, "username" => username))
    end
    logtag("STARTUP", "/set_media $compile_source")
    for m in ["Manga", "Anime"]
        apply_action("set_media", Dict("medium" => m))
    end
end

function compile(port::Integer)
    logtag("STARTUP", "connecting to models")
    models_started = false
    while !models_started
        try
            r = HTTP.get("$MODEL_URL/ready")
            models_started = true
        catch
            logtag("STARTUP", "waiting for models to startup")
            sleep(10)
        end
    end
    logtag("STARTUP", "loading memoize caches")
    get_autocomplete_items_map.([0, 1])
    get_matchedid_map.([0, 1])
    get_images()
    get_missing_images()
    get_media_info.([0, 1])
    Threads.@threads for source in ["mal", "anilist", "kitsu", "animeplanet"]
        compile_source(port, source)
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
