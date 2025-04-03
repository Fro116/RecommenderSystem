module layer4

import CSV
import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/database.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

const PORT = parse(Int, ARGS[1])
const LAYER_3_URL = ARGS[2] != "nothing" ? ARGS[2] : nothing

Oxygen.@post "/read" function read_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    source = data["source"]
    username = data["username"]
    tasks = []
    for table in ["collect_users", "inference_users"]
        task = Threads.@spawn begin
            df = with_db(:inference_read, 3) do db
                query = "SELECT * FROM $table WHERE (source, lower(username)) = (\$1, lower(\$2))"
                stmt = db_prepare(db, query)
                DataFrames.DataFrame(LibPQ.execute(stmt, (source, username)))
            end
        end
        push!(tasks, task)
    end
    user = nothing
    for task in tasks
        df = fetch(task)
        if df isa Symbol || DataFrames.nrow(df) == 0
            continue
        end
        if isnothing(user) || df[:, "db_refreshed_at"] > user[:, "db_refreshed_at"]
            user = df
        end
    end
    if isnothing(user)
        return HTTP.Response(404, [])
    end
    d = Dict(k => only(user[:, k]) for k in DataFrames.names(user))
    HTTP.Response(200, encode(d, :msgpack)...)
end

Oxygen.@post "/write" function write_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    source = data["source"]
    username = data["username"]
    coerce(x) = isnothing(x) ? missing : x
    vals = (
        source,
        username,
        coerce(data["userid"]),
        data["fingerprint"],
        bytes2hex(Vector{UInt8}(data["data"])),
        time()
    )
    r = with_db(:inference_write, 3) do db
        query = "DELETE FROM inference_users WHERE (source, lower(username)) = (\$1, lower(\$2))"
        stmt = db_prepare(db, query)
        LibPQ.execute(stmt, (source, username), binary_format=true)
        query = """
            INSERT INTO inference_users (source, username, userid, fingerprint, data, db_refreshed_at)
            VALUES (\$1, \$2, \$3, \$4, decode(\$5, 'hex'), \$6)
            """
        stmt = db_prepare(db, query)
        LibPQ.execute(stmt, vals, binary_format=true)
    end
    if r isa Symbol
        return HTTP.Response(500, [])
    end
    HTTP.Response(200, [])
end

Oxygen.@post "/fingerprint" function fingerprint_user(r::HTTP.Request)::HTTP.Response
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
    source = data["source"]
    username = data["username"]
    userid = data["userid"]
    d = Dict(source => source, "username" => username, "userid" => userid)
    r = HTTP.post(
        "$LAYER_3_URL/$(source)_fingerprint",
        encode(d, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r)
        return HTTP.Response(r.status, [])
    end
    HTTP.Response(200, r.headers, r.body)
end

function normalize(d::AbstractDict)
    norm(x::AbstractDict) = JSON3.write(x)
    norm(x::AbstractVector) = JSON3.write(x)
    norm(::Nothing) = missing
    norm(x) = x
    Dict(k => norm(v) for (k, v) in d)
end
normalize(d::AbstractVector) = [normalize(x) for x in d]

Oxygen.@post "/fetch" function fetch_user(r::HTTP.Request)::HTTP.Response
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
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
    d = decode(r)
    for k in keys(d)
        d[k] = normalize(d[k])
    end
    HTTP.Response(200, encode(d, :msgpack)...)
end

Oxygen.@post "/refresh" function refresh_user(r::HTTP.Request)::HTTP.Response
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
    source = data["source"]
    username = data["username"]
    force = get(data, "force", false)
    r_read = HTTP.post(
        "http://localhost:$PORT/read",
        encode(Dict("source" => source, "username" => username), :msgpack)...,
        status_exception = false,
    )
    if !HTTP.iserror(r_read)
        d_read = decode(r_read)
        r_fingerprint = HTTP.post(
            "http://localhost:$PORT/fingerprint",
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
        if d_read["fingerprint"] == d_fingerprint["fingerprint"] && !force
            return HTTP.Response(200, encode(Dict("refresh" => false), :msgpack)...)
        end
    else
        r_fingerprint = nothing
    end
    r_fetch = HTTP.post(
        "http://localhost:$PORT/fetch",
        encode(Dict("source" => source, "username" => username), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_fetch)
        return HTTP.Response(r_fetch.status, [])
    end
    d_fetch = decode(r_fetch)
    if isnothing(r_fingerprint)
        r_fingerprint = HTTP.post(
            "http://localhost:$PORT/fingerprint",
            encode(
                Dict(
                    "source" => source,
                    "username" => username,
                    "userid" => d_fetch["usermap"]["userid"],
                ),
                :msgpack,
            )...,
            status_exception = false,
        )
        if HTTP.iserror(r_fingerprint)
            return HTTP.Response(r_fingerprint.status, [])
        end
        d_fingerprint = decode(r_fingerprint)
    end
    data = merge(
        d_fetch["usermap"],
        d_fingerprint,
        Dict(
            "source" => source,
            "data" => CodecZstd.transcode(
                CodecZstd.ZstdCompressor,
                Vector{UInt8}(MsgPack.pack(d_fetch)),
            ),
        ),
    )
    r_write = HTTP.post(
        "http://localhost:$PORT/write",
        encode(data, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_write)
        return HTTP.Response(r_write.status, [])
    end
    HTTP.Response(200, encode(Dict("refresh" => true), :msgpack)...)
end

Oxygen.@post "/autocomplete" function autocomplete(r::HTTP.Request)::HTTP.Response
    data = decode(r)
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

function compile(port::Integer)
    profiles = CSV.read("../../secrets/test.users.csv", DataFrames.DataFrame, stringtype = String)
    @sync for (source, username, userid) in zip(profiles.source, profiles.username, profiles.userid)
        Threads.@spawn begin
            logtag("STARTUP", "/autcomplete")
            HTTP.post(
                "http://localhost:$PORT/autocomplete",
                encode(Dict("source" => source, "prefix" => lowercase(username), "type" => "user"), :msgpack)...,
                status_exception = false,
            )
            logtag("STARTUP", "/read")
            HTTP.post(
                "http://localhost:$PORT/read",
                encode(Dict("source" => source, "username" => username), :msgpack)...,
                status_exception = false,
            )
            logtag("STARTUP", "/refresh")
            HTTP.post(
                "http://localhost:$PORT/refresh",
                encode(Dict("source" => source, "username" => username, "force" => true), :msgpack)...,
                status_exception = false,
            )
        end
    end
end

include("../julia_utils/start_oxygen.jl")

end
