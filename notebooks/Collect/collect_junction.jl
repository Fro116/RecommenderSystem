import Random
include("database.jl")
include("../julia_utils/hash.jl")
include("../julia_utils/http.jl")

PRIORITY_VALS::Set = Set()
PRIORITY_MAXID::Int = 0
const PRIORITY_LOCK = ReentrantLock()

function monitor(primary_table, ts_col)
    df = db_monitor_junction_table(primary_table, ts_col)
    args = ["$name: $val" for (name, val) in zip(df.name, df.value)]
    logtag("MONITOR", join(args, ", "))
end

function prioritize(primary_table::String, idcols::Vector{String}, tscol::String, partitions::Int, N::Int)
    vals = db_prioritize_junction_table(primary_table, idcols, tscol, partitions * N)
    if idcols == ["userid"]
        maxid = db_get_maxid(primary_table, only(idcols))
    end
    lock(PRIORITY_LOCK) do
        global PRIORITY_VALS
        global PRIORITY_MAXID
        idvals = vals
        PRIORITY_VALS = Set([[idvals[i, x] for x in idcols] for i in 1:DataFrames.nrow(idvals)])
        if idcols == ["userid"]
            PRIORITY_MAXID = maxid
        end
    end
end

function garbage_collect(
    primary_table,
    junction_table,
    source_table,
    idcols,
    ts_col,
    source_key,
)
    if idcols == ["userid"]
        while db_insert_missing(primary_table, only(idcols), 1000)
        end
    end
    if !isnothing(source_table)
        if !isnothing(source_key)
            while db_sync_entries(primary_table, junction_table, source_table, idcols, source_key, 1000)
            end
        else
            db_sync_entries(primary_table, junction_table, source_table, idcols, 1000)
        end
    end
    while db_gc_junction_table(primary_table, junction_table, idcols, 1000)
    end
end

function save_entry(
    primary_table::String,
    primary_key::String,
    junction_table::String,
    junction_key::String,
    source_table::Union{String,Nothing},
    source_key::Union{String,Nothing},
    api::String,
    idcols::Vector{String},
    idvals::Vector,
    save_failure::Bool,
)
    r = HTTP.post(api, encode(Dict(idcols .=> idvals), :json)..., status_exception = false)
    if HTTP.iserror(r)
        success = false
        primary_data = nothing
        junction_data = nothing
        if !save_failure
            return
        end        
    else
        success = true
        data = decode(r)
        primary_data = data[primary_key]
        junction_data = data[junction_key]
    end
    db_update_junction_table(
        primary_table,
        junction_table,
        idcols,
        idvals,
        primary_data,
        junction_data,
        success,
    )
    if success && !isnothing(source_key)
        source_data = data[source_key]
        source_idcol = source_key
        if !isnothing(source_data[source_idcol])
            db_update_single_table(
                source_table,
                source_idcol,
                source_data[source_idcol],
                source_data,
                success,
            )
        end
    end
end

function refresh(
    primary_table::String,
    primary_key::String,
    junction_table::String,
    junction_key::String,
    source_table::Union{String,Nothing},
    source_key::Union{String,Nothing},
    api::String,
    idcols::Vector{String},
    tscol::String,
    part::Int,
    partitions::Int,
)
    @assert 0 <= part < partitions
    idvals = Set()
    save(idvals, success) = save_entry(
        primary_table,
        primary_key,
        junction_table,
        junction_key,
        source_table,
        source_key,
        api,
        idcols,
        idvals,
        success,
    )
    while true
        idvals, maxid = lock(PRIORITY_LOCK) do
            Set(x for x in PRIORITY_VALS if (shahash(x) % partitions) == part && x ∉ idvals), PRIORITY_MAXID
        end
        if isempty(idvals)
            logtag("REFRESH", "$part waiting for prioritization")
            sleep(60)
            continue
        end
        for x in Random.shuffle(collect(idvals))
            save(x, true)
        end
        if idcols == ["userid"]
            save([rand([x for x in maxid+1:maxid+10000 if (shahash(x) % partitions) == part])], false)
        end
    end
end

const PRIMARY_TABLE = ARGS[1]
const PRIMARY_KEY = ARGS[2]
const JUNCTION_TABLE = ARGS[3]
const JUNCTION_KEY = ARGS[4]
const SOURCE_TABLE = ARGS[5] == "nothing" ? nothing : ARGS[5]
const SOURCE_KEY = ARGS[6] == "nothing" ? nothing : ARGS[6]
const IDCOLS = Vector{String}(split(ARGS[7], ","))
const TSCOL = ARGS[8]
const API = ARGS[9]
const PARTITIONS = parse(Int, ARGS[10])

prioritize(PRIMARY_TABLE, IDCOLS, TSCOL, PARTITIONS, 600)
@sync begin
    Threads.@spawn @periodic "MONITOR" 600 @handle_errors monitor(PRIMARY_TABLE, TSCOL)
    Threads.@spawn @periodic "PRIORITIZE" 600 @handle_errors prioritize(PRIMARY_TABLE, IDCOLS, TSCOL, PARTITIONS, 600)
    Threads.@spawn @periodic "GARBAGE_COLLECT" 86400 @handle_errors garbage_collect(
        PRIMARY_TABLE,
        JUNCTION_TABLE,
        SOURCE_TABLE,
        IDCOLS,
        TSCOL,
        SOURCE_KEY,
    )
    for i in 1:PARTITIONS
        Threads.@spawn @uniform_delay 600 @handle_errors refresh(
            PRIMARY_TABLE,
            PRIMARY_KEY,
            JUNCTION_TABLE,
            JUNCTION_KEY,
            SOURCE_TABLE,
            SOURCE_KEY,
            API,
            IDCOLS,
            TSCOL,
            i-1,
            PARTITIONS,
        )
    end
end
