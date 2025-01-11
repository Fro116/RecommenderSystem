include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function write_db(key::String, value::Vector{UInt8}, ts::Float64)
    with_db(:update) do db
        query = """
            INSERT INTO external_dependencies (key, value, db_last_success_at)
            VALUES (\$1, decode(\$2, 'hex'), \$3) ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, db_last_success_at = EXCLUDED.db_last_success_at
            """
        stmt = db_prepare(db, query)
        vals = (key, bytes2hex(value), ts)
        LibPQ.execute(stmt, vals; binary_format = true)
    end
end

function save_entry(key::String, url::String)
    for delay in ExponentialBackOff(;n=3)
        r = HTTP.get(url; status_exception=false)
        if HTTP.iserror(r)
            continue
        end
        value = CodecZstd.transcode(CodecZstd.ZstdCompressor, r.body)
        write_db(key, value, time())
        break
    end
end

Threads.@spawn @periodic "MANAMI" 86400 @handle_errors save_entry(
    "manami",
    "https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database.json",
)
