import CSV
import DataFrames
import Glob
import Memoize: @memoize
import ProgressMeter
import ProgressMeter: @showprogress

include("../../julia_utils/http.jl")
include("../../julia_utils/database.jl")
include("../../julia_utils/multithreading.jl")
include("../../julia_utils/scheduling.jl")
include("../../julia_utils/stdout.jl")
const envdir = "../../../environment"
const datadir = "../../../data/lists"
const dbschema = "../../Collect/schema.txt"

function download_users(source::String)
    mkdir("$datadir/$source")
    retrieval = read("$envdir/database/retrieval.txt", String)
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

function get_fingerprints(source::String, items::DataFrames.DataFrame)
    fingerprints = []
    if source in ["mal", "anilist"]
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) == 0
                continue
            end
            if source == "mal"
                update_col = "updated_at"
            elseif source == "anilist"
                update_col = "updatedat"
            else
                @assert false
            end
            d = last(sort(sdf, update_col))
            d = Dict(
                "version" => d["version"],
                "medium" => d["medium"],
                "itemid" => d["itemid"],
                "updated_at" => d[update_col],
            )
            push!(fingerprints, d)
        end
    elseif source == "kitsu"
        if DataFrames.nrow(items) != 0
            update_col = "updatedat"
            d = last(sort(items, update_col))
            d = Dict("version" => d["version"], "updated_at" => d[update_col])
            push!(fingerprints, d)
        end
    elseif source == "animeplanet"
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) != 0
                d = last(sort(sdf, :item_order))
                d = Dict(
                    "version" => d["version"],
                    "medium" => d["medium"],
                    "itemid" => d["itemid"],
                )
                push!(fingerprints, d)
            end
            d = Dict("$(m)_count" => DataFrames.nrow(sdf))
            push!(fingerprints, d)
        end
    end
    fingerprints
end

function save_fingerprints(source::String)
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
    users = CSV.read(
        "$datadir/$source/$(source)_users.csv",
        DataFrames.DataFrame,
        stringtype = String,
        ntasks = 1,
        types = get_typemap(dbschema)["$(source)_users"]
    )
    user_cols = [x for x in DataFrames.names(users) if !startswith(x, "db_")]
    function writecsv(c::Channel)
        p = ProgressMeter.ProgressUnknown(desc = "$source fingerprints"; showspeed=true)
        function write!(dfs, part)
            CSV.write(
                "$datadir/$source.$part.csv",
                reduce(vcat, dfs);
                bufsize = 2^24,
                quotestrings=true,
                transform=(col, val) -> something(val, missing),
            )
            empty!(dfs)
        end
        part = 1
        dfs = []
        try
            while true
                push!(dfs, take!(c))
                if length(dfs) == 1_000_000
                    write!(dfs, part)
                    ProgressMeter.next!(p)
                    part += 1
                end
            end
        catch
        finally
            if length(dfs) > 0
                write!(dfs, part)
            end
        end
        ProgressMeter.finish!(p)
    end
    ch = Channel(writecsv, 1000)
    Threads.@threads for i = 1:DataFrames.nrow(users)
        if ismissing(users[i, :db_last_success_at])
            continue
        end
        if source in ["mal", "animeplanet"]
            if ismissing(users[i, :username])
                continue
            end
            uid = users[i, :username]
            username = users[i, :username]
        elseif source == "anilist"
            if ismissing(users[i, :username]) || ismissing(users[i, :userid])
                continue
            end
            uid = users[i, :userid]
            username = users[i, :username]
        elseif source == "kitsu"
            if ismissing(users[i, :name]) || ismissing(users[i, :userid])
                continue
            end
            uid = users[i, :userid]
            username = users[i, :name]
        else
            @assert false
        end
        userid = users[i, :userid]
        user_data = Dict(k => users[i, k] for k in user_cols)
        db_refreshed_at = users[i, :db_refreshed_at]
        if uid ∉ keys(chunk_map)
            items = []
            fingerprints = []
        else
            items = []
            dfs = []
            for f in chunk_map[uid]
                df = CSV.read(
                    f,
                    DataFrames.DataFrame,
                    stringtype = String,
                    ntasks = 1,
                    types = get_typemap(dbschema)["$(source)_user_items"]
                )
                cols = DataFrames.names(df)
                for i = 1:DataFrames.nrow(df)
                    d = Dict(k => df[i, k] for k in cols)
                    push!(items, d)
                end
                push!(dfs, df)
            end
            df = vcat(dfs...)
            fingerprints = get_fingerprints(source, df)
        end
        user = Dict(
            "user" => user_data,
            "items" => items,
            "usermap" => Dict("username" => username, "userid" => userid),
        )
        df = DataFrames.DataFrame(
            "source" => source,
            "username" => username,
            "userid" => userid,
            "fingerprint" => JSON3.write(fingerprints),
            "data" =>
                "\\x" * bytes2hex(
                    CodecZstd.transcode(
                        CodecZstd.ZstdCompressor,
                        Vector{UInt8}(MsgPack.pack(user)),
                    ),
                ),
            "db_refreshed_at" => db_refreshed_at,
        )
        put!(ch, df)
    end
    close(ch)
    rm("$datadir/$source", force = true, recursive = true)
end

function upload_fingerprints()
    rm(datadir, force = true, recursive = true)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    mkpath(datadir)
    retrieval = read("$envdir/database/retrieval.txt", String)
    cmd = "$retrieval/latest $datadir/latest"
    run(`sh -c $cmd`)
    for source in reverse(sources)
        download_users(source)
        partition(source)
        save_fingerprints(source)
    end
    files = join(readdir(datadir), " ")
    cmd = "cd $datadir && mlr --csv cat *.csv > fingerprints.csv && rm $files"
    run(`sh -c $cmd`)
    save_template = read("$envdir/database/storage.txt", String)
    cmd = replace(save_template, "{INPUT}" => "$datadir/fingerprints.csv", "{OUTPUT}" => "fingerprints.csv")
    run(`sh -c $cmd`)
    cmd = "$envdir/database/import_csv.sh $datadir"
    run(`sh -c $cmd`)
    conn_str = read("$DB_PATH/primary.txt", String)
    cmd = """psql "$conn_str" -f import_csv.sql"""
    run(`sh -c $cmd`)
end

@scheduled "BACKUP" "2:00" @handle_errors upload_fingerprints()
