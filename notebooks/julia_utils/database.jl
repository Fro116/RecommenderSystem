import LibPQ

include("stdout.jl")

const DB_PATH = "../../environment/database"

function get_db_connection()
    conn_str = read("$DB_PATH/primary.txt", String)
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

struct Database
    conn::LibPQ.Connection
    lock::ReentrantLock
    prepared_statements::Dict{String,LibPQ.Statement}
end

const DATABASES = Dict()
const DATABASE_LOCK = ReentrantLock()

function with_db(f, conntype::Symbol)
    db = lock(DATABASE_LOCK) do
        if conntype ∉ keys(DATABASES)
            DATABASES[conntype] = Database(get_db_connection(), ReentrantLock(), Dict())
        end
        DATABASES[conntype]
    end
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
