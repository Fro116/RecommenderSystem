import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/database.jl")

const PORT = parse(Int, ARGS[1])
const LAYER_3_URL = ARGS[2]

Oxygen.@post "/read" function read_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    source = data["source"]
    username = data["username"]
    tasks = []
    for table in ["collect_users", "inference_users"]
        task = Threads.@spawn begin
            df = with_db(Symbol(table), 3) do db
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
        user = df
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
    r = with_db(:write, 0) do db
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

Oxygen.@post "/fetch" function fetch_user(r::HTTP.Request)::HTTP.Response
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
    @assert !HTTP.hasheader(r, "Content-Encoding")
    HTTP.setheader(r, "Content-Encoding" => "zstd")
    r.body = CodecZstd.transcode(CodecZstd.ZstdCompressor, r.body)
    HTTP.Response(200, r.headers, r.body)
end

Oxygen.serveparallel(; host = "0.0.0.0", port = PORT, access_log = nothing, metrics=false, show_banner=false)
