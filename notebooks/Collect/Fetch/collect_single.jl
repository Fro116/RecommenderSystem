import Random
include("database.jl")
include("http.jl")

PRIORITY_VALS::Set = Set()
PRIORITY_MAXID::Int = 0
const PRIORITY_LOCK = ReentrantLock()

function monitor(table::String, idcol::String, secs::Int)
    while true
        curtime = time()
        df = db_monitor_single_table(table, idcol)
        args = ["$name: $val" for (name, val) in zip(df.name, df.value)]
        logtag("MONITOR", join(args, ", "))
        sleep(max(secs - (time() - curtime), 0))
    end
end

function garbage_collect(table::String, idcol::String, secs::Int)
    while true
        curtime = time()
        while db_insert_missing(table, idcol, 100)
        end
        while db_gc_single_table(table, idcol, 100)
        end
        sleep_secs = secs - (time() - curtime)
        if sleep_secs < 0
            logtag("GARBAGE_COLLECT", "late by $sleep_secs seconds")
        else
            sleep(sleep_secs)
        end
    end
end

function prioritize(table::String, idcol::String, secs::Int, partitions::Int)
    while true
        curtime = time()
        vals = Set(db_prioritize_single_table(table, idcol, 100 * partitions)[:, idcol])
        maxid = db_get_maxid(table, idcol)
        lock(PRIORITY_LOCK) do
            global PRIORITY_VALS
            global PRIORITY_MAXID
            PRIORITY_VALS = vals
            PRIORITY_MAXID = maxid
        end
        sleep_secs = secs - (time() - curtime)
        if sleep_secs < 0
            logtag("PRIORITIZE", "late by $sleep_secs seconds")
        else
            sleep(sleep_secs)
        end
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

function refresh(table::String, api::String, idcol::String, part::Int, partitions::Int)
    @assert 0 <= part < partitions
    idvals = Set()
    save(idval, success) = save_entry(table, api, idcol, idval, success)
    while true
        idvals, maxid = lock(PRIORITY_LOCK) do
            Set(x for x in PRIORITY_VALS if (hash(x) % partitions) == part && x ∉ idvals), PRIORITY_MAXID
        end
        if isempty(idvals)
            logtag("REFRESH", "$part waiting for prioritization")
            sleep(10)
            continue
        end
        logtag("REFRESH", "$part with $(length(idvals)) ids")
        for x in Random.shuffle(collect(idvals))
            save(x, true)
        end
        save(rand([x for x in maxid+1:maxid+10000 if (hash(x) % partitions) == part]), false)
    end
end

const TABLE = ARGS[1]
const IDCOL = ARGS[2]
const API = ARGS[3]
const PARTITIONS = parse(Int, ARGS[4])

@sync begin
    Threads.@spawn @handle_errors monitor(TABLE, IDCOL, 60)
    Threads.@spawn @handle_errors prioritize(TABLE, IDCOL, 60, PARTITIONS)
    Threads.@spawn @handle_errors garbage_collect(TABLE, IDCOL, 600)
    for i in 1:PARTITIONS
        Threads.@spawn @handle_errors refresh(TABLE, API, IDCOL, i-1, PARTITIONS)
    end
end
