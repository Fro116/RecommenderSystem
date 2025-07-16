module database

import CSV
import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/database.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
include("../Import/lists/import_history.jl")

const PORT = parse(Int, ARGS[1])
const LAYER_3_URL = ARGS[2] != "nothing" ? ARGS[2] : nothing

Oxygen.@post "/read_user_history" read_user_history(r::HTTP.Request)::HTTP.Response = read_user_history(decode(r))
function read_user_history(data::Dict)::HTTP.Response
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
        return HTTP.Response(404, [])
    end
    df = reduce(vcat, dfs)
    sort!(df, :db_refreshed_at, rev=true)
    d = Dict(k => df[1, k] for k in DataFrames.names(df))
    HTTP.Response(200, encode(d, :msgpack)...)
end

function write_user_history(data::Dict)::HTTP.Response
    source = data["source"]
    username = data["username"]
    coerce(x) = isnothing(x) ? missing : x
    vals = (
        source,
        username,
        coerce(data["userid"]),
        bytes2hex(Vector{UInt8}(data["data"])),
        time()
    )
    r = with_db(:inference_write, 3) do db
        query = """
            INSERT INTO online_user_histories (source, username, userid, data, db_refreshed_at)
            VALUES (\$1, \$2, \$3, decode(\$4, 'hex'), \$5)
            ON CONFLICT (source, lower(username), coalesce(userid, -1))
            DO UPDATE SET source = EXCLUDED.source, username = EXCLUDED.username,
            userid = EXCLUDED.userid, data = EXCLUDED.data, db_refreshed_at = EXCLUDED.db_refreshed_at;
            """
        stmt = db_prepare(db, query)
        LibPQ.execute(stmt, vals, binary_format=true)
    end
    if r isa Symbol
        return HTTP.Response(500, [])
    end
    HTTP.Response(200, [])
end

function fingerprints_match(source::String, user, fingerprint)
    if source == "mal"
        cols = ["version", "medium", "itemid", "updated_at"]
    elseif source == "anilist"
        cols = ["version", "medium", "itemid", "updatedat"]
    elseif source == "kitsu"
        cols = ["version", "updatedat"]
    elseif source == "animeplanet"
        cols = ["version", "medium", "itemid", "item_order"]
    else
        @assert false
    end
    # filter to most recent list
    max_history_tag = nothing
    for x in user["items"]
        tag = x["history_tag"]
        if tag in ["infer", "delete"]
            continue
        end
        if isnothing(max_history_tag) || tag > max_history_tag
            max_history_tag = tag
        end
    end
    items = [x for x in user["items"] if x["history_tag"] == max_history_tag]
    if isempty(fingerprint)
        return isempty(items)
    end
    # check for matches
    for x in fingerprint
        found_match = false
        for y in items
            if all(x[k] == y[k] for k in cols)
                found_match = true
                break
            end
        end
        if !found_match
            return false
        end
    end
    true
end

function normalize(d::AbstractDict)
    norm(x::AbstractDict) = JSON3.write(x)
    norm(x::AbstractVector) = JSON3.write(x)
    norm(::Nothing) = nothing
    norm(::Missing) = nothing
    norm(x) = x
    Dict(lowercase(k) => norm(v) for (k, v) in d)
end
normalize(d::AbstractVector) = [normalize(x) for x in d]

function decompress(x::Vector)
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(x)))
end

Oxygen.@post "/fetch_user_history" fetch_user_history(r::HTTP.Request)::HTTP.Response = fetch_user_history(decode(r))
function fetch_user_history(data::Dict)::HTTP.Response
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    # get user
    source = data["source"]
    username = data["username"]
    if source in ["mal", "animeplanet"]
        args = Dict("username" => username)
    elseif source in ["anilist", "kitsu"]
         r = HTTP.post(
            "$LAYER_3_URL/$(source)_userid",
            encode(Dict("username" => username), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r)
            return HTTP.Response(r.status, [])
        end
        args = Dict("userid" => decode(r)["userid"])
    end
    r = HTTP.post(
        "$LAYER_3_URL/$(source)_user_parallel",
        encode(args, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r)
        return HTTP.Response(r.status, [])
    end
    user = decode(r)
    for k in keys(user)
        user[k] = normalize(user[k])
    end
    # get history
    r_history = read_user_history(
        Dict(
            "source" => data["source"],
            "username" => data["username"],
            "tables" => ["user_histories"],
        )
    )
    if HTTP.iserror(r_history)
        prev_hist = nothing
    else
        prev_hist = decompress(decode(r_history)["data"])
    end
    history = histories.update_history(
        prev_hist,
        user,
        data["source"],
        time(),
        Dates.format(Dates.now(), "yyyymmdd"),
    )
    HTTP.Response(200, encode(history, :msgpack)...)
end

Oxygen.@post "/refresh_user_history" function refresh_user_history(r::HTTP.Request)::HTTP.Response
    # unpack
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
    source = data["source"]
    username = data["username"]
    force_refresh = get(data, "force", false)

    # check if the stored version is already up-to-date
    r_read = read_user_history(Dict("source" => source, "username" => username))
    if !HTTP.iserror(r_read)
        d_read = decode(r_read)
        r_fingerprint = HTTP.post(
            "$LAYER_3_URL/$(source)_fingerprint",
            encode(
                Dict(
                    "source" => source,
                    "username" => username,
                    "userid" => d_read["userid"],
                ),
                :msgpack,
            )...,
            status_exception = false,
        )
        if HTTP.iserror(r_fingerprint)
            return HTTP.Response(r_fingerprint.status, [])
        end
        d_fingerprint = decode(r_fingerprint)
        user = MsgPack.unpack(
            CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(d_read["data"])),
        )
        if fingerprints_match(source, user, JSON3.read(d_fingerprint["fingerprint"])) && !force_refresh
            return HTTP.Response(200, encode(Dict("refresh" => false), :msgpack)...)
        end
    end

    # refresh the user's history
    r_fetch = fetch_user_history(Dict("source" => source, "username" => username))
    if HTTP.iserror(r_fetch)
        return HTTP.Response(r_fetch.status, [])
    end
    d_fetch = decode(r_fetch)
    data = merge(
        d_fetch["usermap"],
        Dict(
            "source" => source,
            "data" => CodecZstd.transcode(
                CodecZstd.ZstdCompressor,
                Vector{UInt8}(MsgPack.pack(d_fetch)),
            ),
        ),
    )
    r_write = write_user_history(data)
    if HTTP.iserror(r_write)
        return HTTP.Response(r_write.status, [])
    end
    HTTP.Response(200, encode(Dict("refresh" => true), :msgpack)...)
end

function compile(port::Integer)
    profiles = CSV.read("../../secrets/test.users.csv", DataFrames.DataFrame, stringtype = String)
    @sync for (source, username, userid) in zip(profiles.source, profiles.username, profiles.userid)
        Threads.@spawn begin
            logtag("STARTUP", "/refresh_user_history")
            HTTP.post(
                "http://localhost:$PORT/refresh_user_history",
                encode(Dict("source" => source, "username" => username, "force" => true), :msgpack)...,
                status_exception = false,
            )
        end
    end
end

include("../julia_utils/start_oxygen.jl")

end
