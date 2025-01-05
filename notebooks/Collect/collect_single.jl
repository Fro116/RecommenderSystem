import Random
include("database.jl")
include("../julia_utils/hash.jl")
include("../julia_utils/http.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

PRIORITY_VALS::Set = Set()
PRIORITY_MAXID::Int = 0
const PRIORITY_LOCK = ReentrantLock()

function monitor(table::String, idcol::String)
    df = db_monitor_single_table(table, idcol)
    args = ["$name: $val" for (name, val) in zip(df.name, df.value)]
    logtag("MONITOR", join(args, ", "))
end

function prioritize(table::String, idcol::String, partitions::Int, N::Int)
    vals = Set(db_prioritize_single_table(table, idcol, partitions * N)[:, idcol])
    maxid = db_get_maxid(table, idcol)
    lock(PRIORITY_LOCK) do
        global PRIORITY_VALS
        global PRIORITY_MAXID
        PRIORITY_VALS = vals
        PRIORITY_MAXID = maxid
    end
end

function garbage_collect(table::String, idcol::String)
    while db_insert_missing(table, idcol, 1000)
    end
    while db_gc_single_table(table, idcol, 1000)
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
            Set(x for x in PRIORITY_VALS if (shahash(x) % partitions) == part && x ∉ idvals), PRIORITY_MAXID
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
        save(rand([x for x in maxid+1:maxid+10000 if (shahash(x) % partitions) == part]), false)
    end
end

const TABLE = ARGS[1]
const IDCOL = ARGS[2]
const API = ARGS[3]
const PARTITIONS = parse(Int, ARGS[4])

prioritize(TABLE, IDCOL, PARTITIONS, 600)
@sync begin
    Threads.@spawn @periodic "MONITOR" 600 @handle_errors monitor(TABLE, IDCOL)
    Threads.@spawn @periodic "PRIORITIZE" 600 @handle_errors prioritize(TABLE, IDCOL, PARTITIONS, 600)
    Threads.@spawn @periodic "GARBAGE_COLLECT" 86400 @handle_errors garbage_collect(TABLE, IDCOL)
    for i in 1:PARTITIONS
        Threads.@spawn @uniform_delay 600 @handle_errors refresh(TABLE, API, IDCOL, i-1, PARTITIONS)
    end
end
