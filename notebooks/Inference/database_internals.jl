Oxygen.@post "/read_user_history" read_user_history(r::HTTP.Request)::HTTP.Response = read_user_history(decode(r))
function read_user_history(data::Dict)::HTTP.Response
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

Oxygen.@post "/write_user_history" write_user_history(r::HTTP.Request)::HTTP.Response = write_user_history(decode(r))
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
            INSERT INTO inference_users (source, username, userid, data, db_refreshed_at)
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

Oxygen.@post "/read_user_embedding" read_user_embedding(r::HTTP.Request)::HTTP.Response = read_user_embedding(decode(r))
function read_user_embedding(data::Dict)::HTTP.Response
    source = data["source"]
    username = data["username"]
    userid = data["userid"]
    coerce(x) = isnothing(x) ? missing : x
    vals = (
        source,
        username,
        coerce(data["userid"]),
    )
    df = with_db(:inference_read, 3) do db
        query = "SELECT * FROM user_embeddings WHERE (source, lower(username), coalesce(userid, -1)) = (\$1, lower(\$2), coalesce(\$3, -1))"
        stmt = db_prepare(db, query)
        DataFrames.DataFrame(LibPQ.execute(stmt, vals))
    end
    if df isa Symbol || DataFrames.nrow(df) == 0
        return HTTP.Response(404, [])
    end
end

Oxygen.@post "/write_user_embedding" write_user_embedding(r::HTTP.Request)::HTTP.Response = write_user_embedding(decode(r))
function write_user_embedding(data::Dict)::HTTP.Response
    source = data["source"]
    username = data["username"]
    training_tag = parse(Int, data["training_tag"])
    finetune_tag = parse(Int, data["finetune_tag"])
    coerce(x) = isnothing(x) ? missing : x
    vals = (
        source,
        username,
        coerce(data["userid"]),
        training_tag,
        finetune_tag,
        bytes2hex(Vector{UInt8}(data["data"])),
    )
    r = with_db(:inference_write, 3) do db
        query = """
            INSERT INTO user_embeddings (source, username, userid, training_tag, finetune_tag, data)
            VALUES (\$1, \$2, \$3, \$4, \$5, decode(\$6, 'hex'))
            ON CONFLICT (source, lower(username), coalesce(userid, -1))
            DO UPDATE SET source = EXCLUDED.source, username = EXCLUDED.username,
            userid = EXCLUDED.userid, training_tag = EXCLUDED.training_tag,
            finetune_tag = EXCLUDED.finetune_tag, data = EXCLUDED.data;
            """
        stmt = db_prepare(db, query)
        LibPQ.execute(stmt, vals, binary_format=true)
    end
    if r isa Symbol
        return HTTP.Response(500, [])
    end
    HTTP.Response(200, [])
end
