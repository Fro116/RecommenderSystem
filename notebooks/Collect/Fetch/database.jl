import Dates
import LibPQ
import SHA
import JSON3

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

function logerror(x::String)
    lock(STDOUT_LOCK) do
        println("$(Dates.now()) [ERROR] $x")
    end
end

function loginfo(x::String)
    lock(STDOUT_LOCK) do
        println("$(Dates.now()) [INFO] $x")
    end
end

const STDOUT_LOCK = ReentrantLock()
const DB_CONNECTION = get_db_connection()
const DB_CONNECTION_LOCK = ReentrantLock()
const DB_PREPARED_STATEMENTS = Dict{String, LibPQ.Statement}()

function with_db_connection(f)
    lock(DB_CONNECTION_LOCK) do
        try
            LibPQ.execute(DB_CONNECTION, "BEGIN;")
            x = f(DB_CONNECTION)
            LibPQ.execute(DB_CONNECTION, "END;")
            return x
        catch e
            logerror("with_db_connection connection error $e")
            LibPQ.reset!(DB_CONNECTION; throw_error = false)
            empty!(DB_PREPARED_STATEMENTS)
        end
    end
end

function db_prepare(conn, query)
    if query ∉ keys(DB_PREPARED_STATEMENTS)
        DB_PREPARED_STATEMENTS[query] = LibPQ.prepare(conn, query)
    end
    return DB_PREPARED_STATEMENTS[query]
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
    conn::LibPQ.Connection,
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
    stmt = db_prepare(conn,
        """
        INSERT INTO $table ($col_str) VALUES ($placeholders)
        ON CONFLICT ($idcol_str) DO UPDATE SET $update_assignments;
        """
    )
    LibPQ.execute(stmt, vals; binary_format=true)
end;

function db_update_single_table(
    table::String,
    idcol::String,
    idval::Union{String,Int},
    data::Dict,
    success::Bool,
)
    with_db_connection() do conn
        curtime = time()
        if !success
            failure_data = Dict(
                idcol => idval,
                "db_refreshed_at" => curtime,
                "db_consecutive_failures" => 1,
            )
            db_upsert(
                conn,
                table,
                [idcol],
                failure_data,
                Dict(
                    "db_consecutive_failures" => "db_consecutive_failures = $table.db_consecutive_failures + 1",
                ),
            )
            return
        end
        hash = canonical_hash(data)
        let
            col_str = join([idcol, "db_entry_hash"], ", ")
            stmt = db_prepare(conn,
                """
                UPDATE $table
                SET db_refreshed_at = \$1,
                db_last_success_at = \$1,
                db_consecutive_failures = 0
                WHERE ($col_str) = (\$2, \$3);
                """
            )
            refresh = LibPQ.execute(stmt, (curtime, idval, hash); binary_format=true)
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
        db_upsert(conn, table, [idcol], merge(data, db_metadata))
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
    with_db_connection() do conn
        curtime = time()
        if !success
            failure_data = Dict(
                (idcols .=> idvals)...,
                "db_refreshed_at" => curtime,
                "db_consecutive_failures" => 1,
            )
            db_upsert(
                conn,
                primary_table,
                idcols,
                failure_data,
                Dict(
                    "db_consecutive_failures" => "db_consecutive_failures = $primary_table.db_consecutive_failures + 1",
                ),
            )
            return
        end
        primary_hash = canonical_hash(primary_data)
        if isnothing(junction_data)
            let
                col_list = [idcols..., "db_primary_hash"]
                col_str = join(col_list, ", ")
                placeholders = join(["\$" * string(i + 1) for i = 1:length(col_list)], ", ")
                stmt = db_prepare(conn,
                    """
                    UPDATE $primary_table
                    SET db_refreshed_at = \$1,
                    db_last_success_at = \$1,
                    db_consecutive_failures = 0
                    WHERE ($col_str) = ($placeholders);
                    """
                )
                refresh = LibPQ.execute(stmt, (curtime, idvals..., primary_hash); binary_format=true)
                if LibPQ.num_affected_rows(refresh) > 0
                    return
                end
            end
            db_metadata = Dict(
                "db_refreshed_at" => curtime,
                "db_primary_last_changed_at" => curtime,
                "db_primary_hash" => primary_hash,
                "db_last_success_at" => curtime,
                "db_consecutive_failures" => 0,
            )
            db_upsert(conn, primary_table, idcols, merge(primary_data, db_metadata))
            return
        end
        junction_hash = canonical_hash(junction_data)
        junction_data = normalize(junction_data)
        let
            col_list = [idcols..., "db_primary_hash", "db_junction_hash"]
            col_str = join(col_list, ", ")
            placeholders = join(["\$" * string(i + 1) for i = 1:length(col_list)], ", ")
            stmt = db_prepare(conn,
                """
                UPDATE $primary_table
                SET db_refreshed_at = \$1,
                db_last_success_at = \$1,
                db_consecutive_failures = 0
                WHERE ($col_str) = ($placeholders);
                """
            )
            refresh = LibPQ.execute(stmt, (curtime, idvals..., primary_hash, junction_hash); binary_format=true)
            if LibPQ.num_affected_rows(refresh) > 0
                return
            end
        end
        db_metadata = Dict(
            "db_refreshed_at" => curtime,
            "db_primary_last_changed_at" => curtime,
            "db_primary_hash" => primary_hash,
            "db_junction_last_changed_at" => curtime,
            "db_junction_hash" => junction_hash,
            "db_last_success_at" => curtime,
            "db_consecutive_failures" => 0,
        )
        db_upsert(conn, primary_table, idcols, merge(primary_data, db_metadata))
        if length(junction_data) > 0
            junction_data = convert(Vector{Dict{String, Any}}, junction_data)
            for x in junction_data
                x["db_junction_last_changed_at"] = curtime
            end
            col_list = collect(keys(first(junction_data)))
            col_str = join(col_list, ", ")
            placeholders = join(["\$" * string(i) for i = 1:length(col_list)], ", ")
            stmt = db_prepare(conn,"INSERT INTO $junction_table ($col_str) VALUES ($placeholders);")
            for x in junction_data
                LibPQ.execute(stmt, Tuple(x[k] for k in col_list); binary_format=true)
            end
        end
    end
end