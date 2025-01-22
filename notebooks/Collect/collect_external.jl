include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function write_db(key::String, data::String)
    ts = time()
    value = CodecZstd.transcode(CodecZstd.ZstdCompressor, data)
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

function save_url(key::String, url::String)
    for delay in ExponentialBackOff(;n=3)
        r = HTTP.get(url; status_exception=false)
        if HTTP.iserror(r)
            continue
        end
        write_db(key, String(r.body))
        break
    end
end

function save_external(key::String, path::String)
    retrieval = read("../../environment/database/retrieval.txt", String)
    datadir = "../../data"
    cmd = "$retrieval/$path $datadir/$path"
    run(`sh -c $cmd`)
    data = read("$datadir/$path", String)
    write_db(key, data)
    rm("$datadir/$path")
end

function save()
    save_url(
        "manami",
        (
            "https://github.com/manami-project/" *
            "anime-offline-database/raw/master/anime-offline-database.json"
        ),
    )
    save_external("media_match_manual", "media_match_manual.csv")
end

@scheduled "EXTERNAL" "00:30" @handle_errors save()
