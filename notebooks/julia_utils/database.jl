import LibPQ

function get_secrets_path()
    for f in ["../../secrets", "../../../secrets"]
        if ispath(f)
            return f
        end
    end
    @assert false
end

function get_db_connection(conntype::Symbol, max_retries::Real)
    if conntype in [:update, :prioritize, :garbage_collect, :monitor, :write]
        dbname = "collect"
    elseif conntype in [:inference_read, :inference_write]
        dbname = "inference"
    else
        @assert false
    end
    conn_strs = readlines("$(get_secrets_path())/db.$dbname.txt")
    retries = 0
    while retries <= max_retries
        retries += 1
        for conn_str in conn_strs
            try
                x = LibPQ.Connection(conn_str)
                @assert LibPQ.status(x) == LibPQ.libpq_c.CONNECTION_OK
                return x
            catch
                nothing
            end
        end
        if retries <= max_retries
            timeout = 1
            logerror("connection failed, retrying in $timeout seconds")
            sleep(timeout)
        end
    end
    logerror("connection failed")
    :connection_failed
end

struct Database
    conn::LibPQ.Connection
    lock::ReentrantLock
    prepared_statements::Dict{String,LibPQ.Statement}
end

const DATABASES = Dict()
const DATABASE_LOCK = ReentrantLock()

function with_db(f, conntype::Symbol, max_retries::Real=Inf)
    db = lock(DATABASE_LOCK) do
        if conntype ∉ keys(DATABASES)
            conn = get_db_connection(conntype, max_retries)
            if conn == :connection_failed
                return conn
            end
            DATABASES[conntype] = Database(conn, ReentrantLock(), Dict())
        end
        DATABASES[conntype]
    end
    if db == :connection_failed
        return db
    end
    lock(db.lock) do
        retries = 0
        while retries <= max_retries
            retries += 1
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
                return :transaction_failed
            end
        end
        logerror("with_db max_retries exceeded")
        :transaction_failed
    end
end

function db_prepare(db::Database, query::String)
    if query ∉ keys(db.prepared_statements)
        db.prepared_statements[query] = LibPQ.prepare(db.conn, query)
    end
    db.prepared_statements[query]
end
