import Random
include("database.jl")
include("http.jl")

function monitor(table, idcol, secs)
    while true
        curtime = time()
        df = db_monitor_single_table(table, idcol)
        args = ["$name: $val," for (name, val) in zip(df.name, df.value)]
        logtag("MONITOR", join(args, " ")[1:end-1])
        while db_insert_missing(table, idcol, 1000)
        end
        while db_gc_single_table(table, idcol, 1000)
        end
        sleep(max(secs - (time() - curtime), 0))
    end
end

function save_entry(table::String, api::String, idcol::String, idval::Int, save_failure::Bool)
    r = HTTP.post(api, encode(Dict(idcol => idval), :json)..., status_exception = false)
    if HTTP.iserror(r)
        success = false
        data = nothing
        if !save_failure
            return
        end
    else
        success = true
        data = decode(r)
    end
    db_update_single_table(table, idcol, idval, data, success)
end

function refresh(table::String, api::String, idcol::String, partition::Tuple{Int,Int})
    while true
        idvals = db_prioritize_single_table(table, idcol, 1000, partition)
        for idval in Random.shuffle(unique(idvals)[:, idcol])
            save_entry(table, api, idcol, idval, true)
        end
        for idval in db_expand_range(table, idcol, 1, partition)[:, idcol]
            save_entry(table, api, idcol, idval, false)
        end
    end
end

const TABLE = ARGS[1]
const IDCOL = ARGS[2]
const API = ARGS[3]
const PARTITIONS = parse(Int, ARGS[4])

@sync begin
    Threads.@spawn monitor(TABLE, IDCOL, 3600)
    for i in 1:PARTITIONS
        Threads.@spawn refresh(TABLE, API, IDCOL, (i-1, PARTITIONS))
    end
end
