include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function write_db(key::String, value::String, ts::Float64)
    with_db(:update) do db
        query = """
            INSERT INTO external_dependencies (key, value, db_last_success_at)
            VALUES (\$1, \$2, \$3) ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, db_last_success_at = EXCLUDED.db_last_success_at
            """
        stmt = db_prepare(db, query)
        vals = (key, value, ts)
        LibPQ.execute(stmt, vals; binary_format = true)
    end
end

function save_entry(key::String, url::String)
    for delay in ExponentialBackOff(;n=3)
        r = HTTP.get(url; status_exception=false)
        if HTTP.iserror(r)
            continue
        end
        try
            value = JSON3.write(JSON3.read(String(r.body)))
            write_db(key, value, time())
            return
        catch e
            logerror("$key received $e when parsing $url")
        end
    end
end

Threads.@spawn @periodic "MANAMI" 86400 @handle_errors save_entry(
    "manami",
    "https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database.json",
)
