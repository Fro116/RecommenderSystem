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

Oxygen.@post "/read_autocomplete" read_autocomplete(r::HTTP.Request)::HTTP.Response = read_autocomplete(decode(r))
function read_autocomplete(data::Dict)::HTTP.Response
    if data["type"] != "user"
        return HTTP.Response(404, [])
    end
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

function get_fingerprints(source::String, items::DataFrames.DataFrame)
    fingerprints = []
    if source in ["mal", "anilist"]
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) == 0
                continue
            end
            if source == "mal"
                update_col = "updated_at"
            elseif source == "anilist"
                update_col = "updatedat"
            else
                @assert false
            end
            d = last(sort(sdf, update_col))
            d = Dict(
                "version" => d["version"],
                "medium" => d["medium"],
                "itemid" => d["itemid"],
                "updated_at" => d[update_col],
            )
            push!(fingerprints, d)
        end
    elseif source == "kitsu"
        if DataFrames.nrow(items) != 0
            update_col = "updatedat"
            d = last(sort(items, update_col))
            d = Dict("version" => d["version"], "updated_at" => d[update_col])
            push!(fingerprints, d)
        end
    elseif source == "animeplanet"
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) != 0
                d = last(sort(sdf, :item_order))
                d = Dict(
                    "version" => d["version"],
                    "medium" => d["medium"],
                    "itemid" => d["itemid"],
                )
                push!(fingerprints, d)
            end
            if DataFrames.nrow(items) > 0
                max_history_tag = items.history_tag[end]
                count = DataFrames.nrow(filter(x -> x.history_tag == max_history_tag, items))
            else
                count = 0
            end
            d = Dict("$(m)_count" => count)
            push!(fingerprints, d)
        end
    end
    fingerprints
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
        item_df = DataFrames.DataFrame([Dict{String,Any}(x) for x in user["items"]])
        d_read_fingerprint = get_fingerprints(source, item_df)
        if d_read_fingerprint == JSON3.read(d_fingerprint["fingerprint"]) && !force_refresh
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
            logtag("STARTUP", "/read_autocomplete")
            HTTP.post(
                "http://localhost:$PORT/read_autocomplete",
                encode(Dict("source" => source, "prefix" => lowercase(username), "type" => "user"), :msgpack)...,
                status_exception = false,
            )
            logtag("STARTUP", "/read_user_history")
            HTTP.post(
                "http://localhost:$PORT/read_user_history",
                encode(Dict("source" => source, "username" => username), :msgpack)...,
                status_exception = false,
            )
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
