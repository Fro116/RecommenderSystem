import CSV
import DataFrames
import Glob
import Memoize: @memoize
import ProgressMeter: @showprogress

include("../../julia_utils/http.jl")
include("../../julia_utils/database.jl")
include("../../julia_utils/multithreading.jl")
include("../../julia_utils/scheduling.jl")
include("../../julia_utils/stdout.jl")
const datadir = "../../../data/import/lists"
const dbschema = "../../Collect/schema.txt"

function download_users(source::String)
    mkdir("$datadir/$source")
    retrieval = "rclone --retries=10 copyto r2:rsys/database/collect"
    tag = read("$datadir/latest", String)
    for t in ["users", "user_items"]
        cmd = "$retrieval/$tag/$(source)_$(t).zstd $datadir/$source/$(source)_$(t).csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $datadir/$source/$(source)_$(t).csv.zstd`)
        rm("$datadir/$source/$(source)_$(t).csv.zstd")
    end
end

@memoize function get_typemap(schema::String)
    dbtypes = Dict(
        "BIGINT" => Int,
        "TEXT" => String,
        "DOUBLE" => Float64,
        "BOOLEAN" => String, # postgres outputs bools as "t", "f"
        "BYTEA" => Vector{UInt8},
    )
    typemap = Dict()
    types = nothing
    table = nothing
    for line in readlines(schema)
        if startswith(line, "CREATE TABLE")
            table = split(line, " ")[3]
            types = Dict()
            continue
        end
        if startswith(line, ");")
            typemap[table] = types
            types = nothing
            table = nothing
            continue
        end
        if !isnothing(table)
            fields = split(strip(replace(line, "," => " ")), " ")
            if fields[1] in ["UNIQUE"]
                continue
            end
            types[lowercase(fields[1])] = dbtypes[fields[2]]
        end
    end
    typemap
end

function get_valid_users(source)
    latest = Dict()
    all_users = Dict()
    to_delete = Set()
    types = get_typemap(dbschema)["$(source)_users"]
    user_cols = [x for x in keys(types) if !startswith(x, "db_")]
    for r in CSV.Rows("$datadir/$source/$(source)_users.csv", types = types, validate=false)
        ts = r.db_last_success_at
        if ismissing(ts)
            continue
        end
        if source in ["mal", "anilist", "animeplanet"]
            username = r.username
            userid = r.userid
        elseif source == "kitsu"
            username = r.name
            userid = r.userid
        else
            @assert false
        end
        user_data = Dict(k => get(r, Symbol(k), nothing) for k in user_cols)
        all_users[(username, userid)] = (user_data, ts)
        k = (lowercase(coalesce(username, "")), coalesce(userid, -1))
        l_ts, l_username, l_userid = get(latest, k, (nothing, nothing, nothing))
        if isnothing(l_ts)
            latest[k] = (ts, username, userid)
        elseif ts > l_ts
            latest[k] = (ts, username, userid)
            push!(to_delete, (l_username, l_userid))
        else
            push!(to_delete, (username, userid))
        end
    end
    logtag("save_lists", "deduping $(length(to_delete)) $source users")
    for k in to_delete
        delete!(all_users, k)
    end
    all_users
end

function partition(source)
    if source in ["mal", "animeplanet"]
        col = :username
    elseif source in ["anilist", "kitsu"]
        col = :userid
    else
        @assert false
    end
    mkdir("$datadir/$source/splits")
    cmd = (
        "mlr --csv split -n 1000000 --prefix $datadir/$source/splits/split " *
        "$datadir/$source/$(source)_user_items.csv"
    )
    run(`sh -c $cmd`)
    rm("$datadir/$source/$(source)_user_items.csv")
    @showprogress for (chunk_idx, f) in
                      Iterators.enumerate(Glob.glob("$datadir/$source/splits/split_*.csv"))
        savedir = "$datadir/$source/user_items/$chunk_idx"
        mkpath(savedir)
        run(`mlr --csv --from $f split -g $(string(col)) --prefix $savedir/ -j ""`)
    end
    rm("$datadir/$source/splits", recursive = true, force = true)
end

function gather(source)
    chunks = readdir("$datadir/$source/user_items")
    chunk_map = Dict()
    for c in chunks
        for f in readdir("$datadir/$source/user_items/$c")
            if source in ["mal", "animeplanet"]
                parser = identity
            elseif source in ["kitsu", "anilist"]
                parser = x -> parse(Int, x)
            else
                @assert false
            end
            user = parser(f[1:end-length(".csv")])
            if user ∉ keys(chunk_map)
                chunk_map[user] = []
            end
            push!(chunk_map[user], "$datadir/$source/user_items/$c/$f")
        end
    end
    types = get_typemap(dbschema)["$(source)_user_items"]
    mkpath("$datadir/$source/lists")
    users = get_valid_users(source)
    batches = Iterators.partition(collect(users), 1_000_000)
    @showprogress for (b, batch) in Iterators.enumerate(batches)
        dfs = Vector{Any}(undef, length(batch))
        Threads.@threads for i = 1:length(batch)
            ((username, userid), (v, ts)) = batch[i]

            if source in ["mal", "animeplanet"]
                k = username
            elseif source in ["anilist", "kitsu"]
                k = userid
            else
                @assert false
            end
            items = []
            for f in get(chunk_map, k, [])
                df = CSV.read(
                    f,
                    DataFrames.DataFrame,
                    stringtype = String,
                    ntasks = 1,
                    types = types,
                )
                for i = 1:DataFrames.nrow(df)
                    d = Dict(k => df[i, k] for k in keys(types))
                    push!(items, d)
                end
            end
            user = Dict(
                "user" => v,
                "items" => items,
                "usermap" => Dict("username" => username, "userid" => userid),
            )
            dfs[i] = DataFrames.DataFrame(
                "source" => source,
                "username" => username,
                "userid" => userid,
                "data" =>
                    "\\x" * bytes2hex(
                        CodecZstd.transcode(
                            CodecZstd.ZstdCompressor,
                            Vector{UInt8}(MsgPack.pack(user)),
                        ),
                    ),
                "db_refreshed_at" => ts,
            )
        end
        CSV.write(
            "$datadir/$source.$b.csv",
            reduce(vcat, dfs);
            bufsize = 2^24,
            quotestrings = true,
            transform = (col, val) -> something(val, missing),
        )
    end
end

function save_lists(datetag)
    rm(datadir, recursive = true, force = true)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    mkpath(datadir)
    open("$datadir/latest", "w") do f
        write(f, datetag)
    end
    for source in reverse(sources)
        download_users(source)
        partition(source)
        gather(source)
        rm("$datadir/$source", recursive = true, force = true)
    end
    tag = read("$datadir/latest", String)
    files = join(readdir(datadir), " ")
    cmds = [
        "cd $datadir",
        "mlr --csv cat *.csv > lists.csv",
        "zstd lists.csv -o lists.csv.zstd",
        "rclone --retries=10 copyto lists.csv.zstd r2:rsys/database/lists/$tag/lists.csv.zstd",
        "rm $files",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

save_lists(ARGS[1])
