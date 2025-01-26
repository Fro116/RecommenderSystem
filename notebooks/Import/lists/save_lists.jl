import CSV
import DataFrames
import Glob

include("../../julia_utils/http.jl")
include("../../julia_utils/database.jl")
include("../../julia_utils/multithreading.jl")
include("../../julia_utils/scheduling.jl")
include("../../julia_utils/stdout.jl")
const envdir = "../../../environment"
const datadir = "../../../data/users"

function download_users(source::String)
    mkdir("$datadir/$source")
    retrieval = read("$envdir/database/retrieval.txt", String)
    cmd = "$retrieval/latest $datadir/$source/latest"
    run(`sh -c $cmd`)
    tag = read("$datadir/$source/latest", String)
    for t in ["users", "user_items"]
        cmd = "$retrieval/$tag/$(source)_$(t).zstd $datadir/$source/$(source)_$(t).csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $datadir/$source/$(source)_$(t).csv.zstd`)
        rm("$datadir/$source/$(source)_$(t).csv.zstd")
    end
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
    mkdir("$datadir/$source/user_items")
    for (chunk_idx, f) in
        Iterators.enumerate(Glob.glob("$datadir/$source/splits/split_*.csv"))
        mkdir("$datadir/$source/user_items/$chunk_idx")
        df = CSV.read(f, DataFrames.DataFrame, ntasks=1)
        gdf = DataFrames.groupby(df, col)
        Threads.@threads for subdf in gdf
            name = first(subdf[:, col])
            filename = "$datadir/$source/user_items/$chunk_idx/$name.csv"
            CSV.write(filename, subdf, quotestrings=true)
        end
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
            if DataFrames.nrow(sdf) == 0
                continue
            end
            d = last(sort(sdf, :item_order))
            d = Dict(
                "version" => d["version"],
                "medium" => d["medium"],
                "itemid" => d["itemid"],
            )
            push!(fingerprints, d)
            d = Dict("version" => d["version"], "$(m)_count" => DataFrames.nrow(sdf))
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
                parser = lowercase
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
    )
    user_cols = [x for x in DataFrames.names(users) if !startswith(x, "db_")]
    ret = Channel(Inf)
    task = Threads.@spawn @handle_errors multithreading.collect(ret)
    Threads.@threads for i = 1:DataFrames.nrow(users)
        if ismissing(users[i, :db_last_success_at])
            continue
        end
        if source in ["mal", "animeplanet"]
            if ismissing(users[i, :username])
                continue
            end
            uid = lowercase(users[i, :username])
            username = lowercase(users[i, :username])
        elseif source == "anilist"
            if ismissing(users[i, :username]) || ismissing(users[i, :userid])
                continue
            end
            uid = users[i, :userid]
            username = lowercase(users[i, :username])
        elseif source == "kitsu"
            if ismissing(users[i, :name]) || ismissing(users[i, :userid])
                continue
            end
            uid = users[i, :userid]
            username = lowercase(users[i, :name])
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
                    typemap = Dict(Dates.Date => String, Dates.Time => String),
                    ntasks = 1,
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
        d = Dict(
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
        put!(ret, d)
    end
    close(ret)
    DataFrames.DataFrame(fetch(task))
end

function upload_fingerprints()
    rm(datadir, force = true, recursive = true)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    mkpath(datadir)
    Threads.@threads for source in sources
        download_users(source)
        partition(source)
        fingerprints = save_fingerprints(source)
        fingerprints = fingerprints[
            :,
            [:source, :username, :userid, :fingerprint, :data, :db_refreshed_at],
        ]
        CSV.write("$datadir/$source.fingerprints.csv", fingerprints; bufsize = 2^24, quotestrings=true)
        rm("$datadir/$source", force = true, recursive = true)
    end
    files = join(["$s.fingerprints.csv" for s in sources], " ")
    cmd = "cd $datadir && mlr --csv cat $files > fingerprints.csv && rm $files"
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

@scheduled "BACKUP" "2:00" upload_fingerprints()
