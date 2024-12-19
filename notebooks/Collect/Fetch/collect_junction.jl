import Random
include("database.jl")
include("http.jl")

function monitor(
    primary_table,
    junction_table,
    source_table,
    idcols,
    ts_col,
    source_key,
    secs,
)
    while true
        curtime = time()
        df = db_monitor_junction_table(primary_table, ts_col)
        args = ["$name: $val," for (name, val) in zip(df.name, df.value)]
        logtag("MONITOR", join(args, " ")[1:end-1])
        if idcols == ["userid"]
            while db_insert_missing(primary_table, only(idcols), 1000)
            end
        end
        if !isnothing(source_table)
            flush(stdout)
            while db_insert_source(primary_table, source_table, idcols, 1000)
                flush(stdout)
            end
            while db_gc_junction_table(
                primary_table,
                junction_table,
                source_table,
                only(idcols),
                source_key,
                1000,
            )
            end
        end
        while db_gc_junction_table(primary_table, junction_table, idcols, 1000)
        end
        sleep(max(secs - (time() - curtime), 0))
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
        db_update_single_table(
            source_table,
            source_idcol,
            source_data[source_idcol],
            source_data,
            success,
        )
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
    partition::Tuple{Int,Int},
)
    while true
        df = db_prioritize_junction_table(primary_table, idcols, tscol, 1000, partition)
        for i = Random.shuffle(1:DataFrames.nrow(df))
            idvals = [df[i, x] for x in idcols]
            save_entry(
                primary_table,
                primary_key,
                junction_table,
                junction_key,
                source_table,
                source_key,
                api,
                idcols,
                idvals,
                true,
            )
        end
        if idcols == ["userid"]
            df = db_expand_range(primary_table, only(idcols), 1, partition)
            for i = 1:DataFrames.nrow(df)
                idvals = [df[i, x] for x in idcols]
                save_entry(
                    primary_table,
                    primary_key,
                    junction_table,
                    junction_key,
                    source_table,
                    source_key,
                    api,
                    idcols,
                    idvals,
                    false,
                )
            end
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

@sync begin
    Threads.@spawn monitor(
        PRIMARY_TABLE,
        JUNCTION_TABLE,
        SOURCE_TABLE,
        IDCOLS,
        TSCOL,
        SOURCE_KEY,
        3600,
    )
    for i in 1:PARTITIONS
        Threads.@spawn refresh(
            PRIMARY_TABLE,
            PRIMARY_KEY,
            JUNCTION_TABLE,
            JUNCTION_KEY,
            SOURCE_TABLE,
            SOURCE_KEY,
            API,
            IDCOLS,
            TSCOL,
            (i-1, PARTITIONS),
        )
    end
end
