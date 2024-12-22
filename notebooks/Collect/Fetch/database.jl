import Dates
import DataFrames
import JSON3
import LibPQ
import SHA

function get_db_connection()
    conn_str = read("../../../environment/database/primary.txt", String)
    while true
        try
            x = LibPQ.Connection(conn_str)
            @assert LibPQ.status(x) == LibPQ.libpq_c.CONNECTION_OK
            return x
        catch
            timeout = 1
            logerror("connection failed, retrying in $timeout seconds")
            sleep(timeout)
        end
    end
end

function logtag(tag::String, x::String)
    lock(STDOUT_LOCK) do
        println("$(Dates.now()) [$tag] $x")
    end
end
logerror(x::String) = logtag("ERROR", x)

const STDOUT_LOCK = ReentrantLock()

macro handle_errors(ex)
    quote
        try
            $(esc(ex))
        catch err
            lock(STDOUT_LOCK) do
                Base.showerror(stdout, err, catch_backtrace())
                println()
            end
        end
    end
end

struct Database
    conn::LibPQ.Connection
    lock::ReentrantLock
    prepared_statements::Dict{String,LibPQ.Statement}
end

const DATABASES = Dict(
    :update => Database(get_db_connection(), ReentrantLock(), Dict()),
    :prioritize => Database(get_db_connection(), ReentrantLock(), Dict()),
    :monitor => Database(get_db_connection(), ReentrantLock(), Dict()),
    :garbage_collect => Database(get_db_connection(), ReentrantLock(), Dict()),
)

function with_db(f, conntype::Symbol)
    db = DATABASES[conntype]
    lock(db.lock) do
        while true
            try
                LibPQ.execute(db.conn, "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                x = f(db)
                LibPQ.execute(db.conn, "END;")
                return x
            catch e
                if e isa LibPQ.Errors.SerializationFailure
                    continue
                end
                logerror("with_db connection error $e")
                LibPQ.reset!(db.conn; throw_error = false)
                empty!(db.prepared_statements)
                return
            end
        end
    end
end

function db_prepare(db::Database, query::String)
    if query ∉ keys(db.prepared_statements)
        db.prepared_statements[query] = LibPQ.prepare(db.conn, query)
    end
    db.prepared_statements[query]
end

function canonical_hash(d::AbstractDict)
    canon(x::AbstractDict) = Dict(k => canon(x[k]) for k in sort(collect(keys(x))))
    canon(x::AbstractVector) = [canon(y) for y in x]
    canon(x) = x
    bytes2hex(SHA.sha256(JSON3.write(canon(d))))
end
function canonical_hash(d::AbstractVector)
    hashes = sort([canonical_hash(x) for x in d])
    bytes2hex(SHA.sha256(JSON3.write(hashes)))
end
function normalize(d::AbstractDict)
    norm(x::AbstractDict) = JSON3.write(x)
    norm(x::AbstractVector) = JSON3.write(x)
    norm(::Nothing) = missing
    norm(x) = x
    Dict(k => norm(v) for (k, v) in d)
end
normalize(d::AbstractVector) = [normalize(x) for x in d]

function db_upsert(
    db::Database,
    table::String,
    idcols::Vector{String},
    data::Dict,
    conflict_rules::Dict = Dict(),
)
    data = normalize(data)
    cols = collect(keys(data))
    vals = Tuple(data[k] for k in cols)
    col_str = join(cols, ", ")
    idcol_str = join(idcols, ", ")
    placeholders = join(["\$" * string(i) for i = 1:length(vals)], ", ")
    conflict_rules =
        merge(Dict(c => c * " = EXCLUDED." * c for c in cols if c ∉ idcols), conflict_rules)
    update_assignments = join(collect(values(conflict_rules)), ", ")
    query = """
        INSERT INTO $table ($col_str) VALUES ($placeholders)
        ON CONFLICT ($idcol_str) DO UPDATE SET $update_assignments;
        """
    stmt = db_prepare(db, query)
    LibPQ.execute(stmt, vals; binary_format = true)
end;

function db_update_single_table(
    table::String,
    idcol::String,
    idval::Union{String,Int},
    data::Union{Dict,Nothing},
    success::Bool,
)
    with_db(:update) do db
        curtime = time()
        if !success
            failure_data = Dict(
                idcol => idval,
                "db_refreshed_at" => curtime,
                "db_consecutive_failures" => 1,
            )
            db_upsert(
                db,
                table,
                [idcol],
                failure_data,
                Dict(
                    "db_consecutive_failures" => "db_consecutive_failures = COALESCE($table.db_consecutive_failures, 0) + 1",
                ),
            )
            return
        end
        hash = canonical_hash(data)
        let
            col_str = join([idcol, "db_entry_hash"], ", ")
            query = """
                UPDATE $table
                SET db_refreshed_at = \$1,
                db_last_success_at = \$1,
                db_consecutive_failures = 0
                WHERE ($col_str) = (\$2, \$3);
                """
            stmt = db_prepare(db, query)
            refresh = LibPQ.execute(stmt, (curtime, idval, hash); binary_format = true)
            if LibPQ.num_affected_rows(refresh) > 0
                return
            end
        end
        db_metadata = Dict(
            "db_refreshed_at" => curtime,
            "db_last_changed_at" => curtime,
            "db_entry_hash" => hash,
            "db_last_success_at" => curtime,
            "db_consecutive_failures" => 0,
        )
        db_upsert(db, table, [idcol], merge(data, db_metadata))
    end
end

function db_prioritize_single_table(
    table::String,
    idcol::String,
    N::Int,
)
    with_db(:prioritize) do db
        query = """
            SELECT $idcol FROM $table
            ORDER BY db_refreshed_at ASC LIMIT \$1;
            """
        stmt = db_prepare(db, query)
        oldest_items = DataFrames.DataFrame(LibPQ.execute(stmt, (N,); binary_format = true))
        query = """
            SELECT $idcol FROM $table
            WHERE db_refreshed_at IS NULL
            ORDER BY RANDOM() LIMIT \$1;
            """
        stmt = db_prepare(db, query)
        new_items = DataFrames.DataFrame(LibPQ.execute(stmt, (N,); binary_format = true))
        unique(vcat(oldest_items, new_items))
    end
end

function db_get_maxid(table::String, idcol::String)
    with_db(:prioritize) do db
        query = """
            SELECT MAX($idcol) AS maxid FROM $table
            WHERE db_consecutive_failures = 0
            """
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt))
        coalesce(df.maxid..., 0)
    end
end

function db_insert_missing(table::String, idcol::String, N::Int)
    with_db(:garbage_collect) do db
        query = """
            SELECT MAX($idcol) AS maxid FROM $table
            WHERE db_consecutive_failures = 0
            """
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt))
        maxid = coalesce(df.maxid..., 0)
        query = """
            SELECT $idcol
            FROM generate_series(1, \$1) AS $idcol
            EXCEPT SELECT $idcol FROM $table
            LIMIT \$2;
            """
        stmt = db_prepare(db, query)
        missing_ids =
            DataFrames.DataFrame(LibPQ.execute(stmt, tuple(maxid, N); binary_format = true))[
                :,
                idcol,
            ]
        query = """
            INSERT INTO $table ($idcol) VALUES (\$1)
            ON CONFLICT DO NOTHING;
            """
        stmt = db_prepare(db, query)
        for x in missing_ids
            LibPQ.execute(stmt, tuple(x); binary_format = true)
        end
        length(missing_ids) == N
    end
end

function db_monitor_single_table(table::String, idcol::String)
    with_db(:monitor) do db
        curtime = time()
        hour = 3600
        day = 24 * hour
        entries = []
        function run(name::String, query::String, params::Union{Tuple,Nothing})
            stmt = db_prepare(db, query)
            if !isnothing(params)
                r = LibPQ.execute(stmt, params; binary_format = true)
            else
                r = LibPQ.execute(stmt)
            end
            df = DataFrames.DataFrame(r)
            df[!, "name"] = [name]
            push!(entries, df)
        end
        run(
            "hourly_success_frac",
            """
            SELECT SUM(CASE WHEN db_consecutive_failures = 0 THEN 1 ELSE 0 END)::float / COUNT(*)
            AS value FROM $table
            WHERE \$1 - db_refreshed_at < \$2
            """,
            tuple(curtime, hour),
        )
        run(
            "hourly_throughput",
            """
            SELECT COUNT(*) AS value FROM $table
            WHERE \$1 - db_refreshed_at < \$2
            """,
            tuple(curtime, hour),
        )
        run(
            "oldest_days",
            """
            SELECT (\$1 - MIN(db_refreshed_at)) / \$2 AS value FROM $table
            """,
            tuple(curtime, day),
        )
        run(
            "max_id",
            """
            SELECT MAX($idcol) AS value FROM $table
            WHERE db_consecutive_failures = 0
            """,
            nothing,
        )
        run(
            "count",
            """
            SELECT COUNT(*) AS value FROM $table
            """,
            nothing,
        )
        vcat(entries...)
    end
end

function db_gc_single_table(table::String, idcol::String, N::Int)
    with_db(:garbage_collect) do db
        curtime = time()
        day = 86400.0
        year = 365 * day
        month = year / 12
        query = """
            WITH todelete AS (
                SELECT $idcol FROM $table
                WHERE db_consecutive_failures >= \$1
                AND \$2 - db_last_success_at >= \$3
                ORDER BY db_last_success_at ASC LIMIT \$4
            )
            DELETE FROM $table WHERE $idcol IN (SELECT $idcol FROM todelete);
        """
        stmt = db_prepare(db, query)
        r = LibPQ.execute(stmt, tuple(3, curtime, month, N); binary_format = true)
        LibPQ.num_affected_rows(r) == N
    end
end

function db_update_junction_table(
    primary_table::String,
    junction_table::String,
    idcols::Vector{String},
    idvals::Vector,
    primary_data::Union{Dict,Nothing},
    junction_data::Union{Vector,Nothing},
    success::Bool,
)
    with_db(:update) do db
        curtime = time()
        if !success
            failure_data = Dict(
                (idcols .=> idvals)...,
                "db_refreshed_at" => curtime,
                "db_consecutive_failures" => 1,
            )
            db_upsert(
                db,
                primary_table,
                idcols,
                failure_data,
                Dict(
                    "db_consecutive_failures" => "db_consecutive_failures = COALESCE($primary_table.db_consecutive_failures, 0) + 1",
                ),
            )
            return
        end
        primary_hash = canonical_hash(primary_data)
        junction_hash = canonical_hash(junction_data)
        junction_data = normalize(junction_data)
        col_str = join(idcols, ", ")
        placeholders = join(["\$" * string(i) for i = 1:length(idcols)], ", ")
        query = """
            SELECT db_primary_hash, db_junction_hash FROM $primary_table
            WHERE ($col_str) = ($placeholders);
            """
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt, (idvals...,); binary_format = true))
        db_metadata = Dict(
            "db_refreshed_at" => curtime,
            "db_primary_last_changed_at" => curtime,
            "db_primary_hash" => primary_hash,
            "db_junction_last_changed_at" => curtime,
            "db_junction_hash" => junction_hash,
            "db_last_success_at" => curtime,
            "db_consecutive_failures" => 0,
        )
        if DataFrames.nrow(df) > 0
            if coalesce(only(df.db_primary_hash), nothing) == primary_hash
                delete!(db_metadata, "db_primary_last_changed_at")
            end
            if coalesce(only(df.db_junction_hash), nothing) == junction_hash
                delete!(db_metadata, "db_junction_last_changed_at")
            end
        end
        db_upsert(db, primary_table, idcols, merge(primary_data, db_metadata))
        let
            col_str = join(idcols, ", ")
            placeholders = join(["\$" * string(i) for i = 1:length(idcols)], ", ")
            query = "DELETE FROM $junction_table WHERE ($col_str) = ($placeholders);"
            stmt = db_prepare(db, query)
            LibPQ.execute(stmt, (idvals...,); binary_format = true)
        end
        if length(junction_data) > 0
            col_list = collect(keys(first(junction_data)))
            col_str = join(col_list, ", ")
            placeholders = join(["\$" * string(i) for i = 1:length(col_list)], ", ")
            query = "INSERT INTO $junction_table ($col_str) VALUES ($placeholders);"
            stmt = db_prepare(db, query)
            for x in junction_data
                LibPQ.execute(stmt, Tuple(x[k] for k in col_list); binary_format = true)
            end
        end
    end
end

function db_prioritize_junction_table(
    primary_table::String,
    idcols::Vector{String},
    tscol::String,
    N::Int,
)
    with_db(:prioritize) do db
        curtime = time()
        entries = []
        function run(query::String, params::Tuple)
            stmt = db_prepare(db, query)
            r = LibPQ.execute(stmt, params; binary_format = true)
            push!(entries, DataFrames.DataFrame(r))
        end
        run(
            """
            SELECT $(join(idcols, ", ")) FROM $primary_table
            WHERE db_refreshed_at IS NULL
            ORDER BY RANDOM() ASC LIMIT \$1;
            """,
            (N,),
        )
        day = 86400.0
        week = 7 * day
        year = 365 * day
        month = year / 12
        quarter = year / 4
        time_query = """
            SELECT $(join(idcols, ", ")) FROM $primary_table
            WHERE db_refreshed_at - $tscol < \$1
            AND \$2 - db_refreshed_at > \$3
            ORDER BY db_refreshed_at ASC LIMIT \$4;
            """
        run(time_query, tuple(week, curtime, day, N))
        run(time_query, tuple(month, curtime, week, N))
        run(time_query, tuple(quarter, curtime, month, N))
        run(time_query, tuple(year, curtime, quarter, N))
        run(
            """
            SELECT $(join(idcols, ", ")) FROM $primary_table
            WHERE (db_consecutive_failures < \$1 OR \$2 - db_last_success_at < \$3)
            ORDER BY db_refreshed_at ASC LIMIT \$4;
            """,
            tuple(3, curtime, month, N),
        )
        run(
            """
            SELECT $(join(idcols, ", ")) FROM $primary_table
            WHERE db_consecutive_failures >= \$1
            AND \$2 - db_last_success_at >= \$3
            AND \$2 - db_refreshed_at > \$4
            ORDER BY db_refreshed_at ASC LIMIT \$5;
            """,
            tuple(3, curtime, month, quarter, N),
        )
        unique(vcat(entries...))
    end
end

function db_monitor_junction_table(primary_table::String, tscol::String)
    with_db(:monitor) do db
        curtime = time()
        hour = 3600
        day = 24 * hour
        week = 7 * day
        year = 365 * day
        month = year / 12
        quarter = year / 4
        entries = []
        function run(name::String, query::String, params::Union{Tuple,Nothing})
            stmt = db_prepare(db, query)
            if !isnothing(params)
                r = LibPQ.execute(stmt, params; binary_format = true)
            else
                r = LibPQ.execute(stmt)
            end
            df = DataFrames.DataFrame(r)
            df[!, "name"] = [name]
            push!(entries, df)
        end
        run(
            "hourly_success_frac",
            """
            SELECT SUM(CASE WHEN db_consecutive_failures = 0 THEN 1 ELSE 0 END)::float / COUNT(*)
            AS value FROM $primary_table
            WHERE \$1 - db_refreshed_at < \$2
            """,
            tuple(curtime, hour),
        )
        run(
            "hourly_throughput",
            """
            SELECT COUNT(*) AS value FROM $primary_table
            WHERE \$1 - db_refreshed_at < \$2
            """,
            tuple(curtime, hour),
        )
        run(
            "oldest_success",
            """
            SELECT (\$1 - MIN(db_refreshed_at)) / \$2 AS value FROM $primary_table
            WHERE db_consecutive_failures = \$3;
            """,
            tuple(curtime, day, 0),
        )
        run(
            "oldest_failure",
            """
            SELECT (\$1 - MIN(db_refreshed_at)) / \$2 AS value FROM $primary_table
            WHERE db_consecutive_failures != \$3;
            """,
            tuple(curtime, day, 0),
        )
        run(
            "new_items",
            """
            SELECT COUNT(*) AS value FROM $primary_table
            WHERE db_refreshed_at IS NULL;
            """,
            nothing,
        )
        time_query = """
            SELECT COUNT(*) AS value FROM $primary_table
            WHERE db_refreshed_at - $tscol < \$1
            AND \$2 - db_refreshed_at > \$3;
            """
        run("daily_refresh_items", time_query, tuple(week, curtime, day))
        run("weekly_refresh_items", time_query, tuple(month, curtime, week))
        run("monthly_refresh_items", time_query, tuple(quarter, curtime, month))
        run("quarterly_refresh_items", time_query, tuple(year, curtime, quarter))
        run(
            "valid_items",
            """
            SELECT COUNT(*) AS value FROM $primary_table
            WHERE db_consecutive_failures < \$1
            OR \$2 - db_last_success_at < \$3;
            """,
            tuple(3, curtime, month),
        )
        run(
            "invalid_items",
            """
            SELECT COUNT(*) AS value FROM $primary_table
            WHERE db_consecutive_failures >= \$1
            AND \$2 - db_last_success_at >= \$3;
            """,
            tuple(3, curtime, month),
        )
        vcat(entries...)
    end
end

function db_sync_entries(
    primary_table::String,
    junction_table::String,
    source_table::String,
    idcols::Vector{String},
    source_key::Union{String,Nothing},
)
    df = with_db(:garbage_collect) do db
        col_str = join(idcols, ", ")
        query = "SELECT DISTINCT $col_str FROM $source_table;"
        stmt = db_prepare(db, query)
        DataFrames.DataFrame(LibPQ.execute(stmt; binary_format = true))
    end
    source_vals = Set(Tuple(df[i, x] for x in idcols) for i in 1:DataFrames.nrow(df))
    with_db(:garbage_collect) do db
        col_str = join(idcols, ", ")
        query = "SELECT DISTINCT $col_str FROM $primary_table;"
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt; binary_format = true))
        existing_vals = Set(Tuple(df[i, x] for x in idcols) for i in 1:DataFrames.nrow(df))
        placeholders = join(["\$" * string(i) for i = 1:length(idcols)], ", ")
        query = "INSERT INTO $primary_table ($col_str) VALUES ($placeholders) ON CONFLICT DO NOTHING;"
        stmt = db_prepare(db, query)
        for x in setdiff(source_vals, existing_vals)
            LibPQ.execute(stmt, x; binary_format = true)
        end
    end
    with_db(:garbage_collect) do db
        col_str = join(idcols, ", ")
        if isnothing(source_key)
            source_key_filter = ""
        else
            source_key_filter = "WHERE $source_key IS NOT NULL"
        end
        query = "SELECT DISTINCT $col_str FROM $primary_table $source_key_filter;"
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt; binary_format = true))
        existing_vals = Set(Tuple(df[i, x] for x in idcols) for i in 1:DataFrames.nrow(df))
        placeholders = join(["\$" * string(i) for i = 1:length(idcols)], ", ")
        for x in setdiff(existing_vals, source_vals)
            for table in [primary_table, junction_table]
                query = "DELETE FROM $table WHERE ($col_str) = ($placeholders);"
                stmt = db_prepare(db, query)
                LibPQ.execute(stmt, x; binary_format = true)
            end
        end
    end
end

function db_gc_junction_table(
    primary_table::String,
    junction_table::String,
    idcols::Vector{String},
    N::Int,
)
    with_db(:garbage_collect) do db
        curtime = time()
        day = 86400.0
        year = 365 * day
        month = year / 12
        col_str = join(idcols, ", ")
        query = """
            SELECT $col_str FROM $primary_table
            WHERE db_consecutive_failures >= \$1
            AND \$2 - db_last_success_at >= \$3
            LIMIT \$4;
            """
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(
            LibPQ.execute(stmt, tuple(3, curtime, month, N); binary_format = true),
        )
        for i = 1:DataFrames.nrow(df)
            vals = [df[i, x] for x in idcols]
            placeholders = join(["\$" * string(i) for i = 1:length(vals)], ", ")
            for table in [primary_table, junction_table]
                query = "DELETE FROM $table WHERE ($col_str) = ($placeholders);"
                stmt = db_prepare(db, query)
                LibPQ.execute(stmt, (vals...,); binary_format = true)
            end
        end
        DataFrames.nrow(df) == N
    end
end