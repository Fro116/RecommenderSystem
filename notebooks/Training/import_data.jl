import CSV
import DataFrames
import Glob
import Memoize: @memoize
import MsgPack
import Random
import ProgressMeter: @showprogress
include("../julia_utils/stdout.jl")
include("import_list.jl")

const MEDIUMS = ["manga", "anime"]
const SOURCES = ["mal", "anilist", "kitsu", "animeplanet"]
const datadir = "../../data/training"
const envdir = "../../environment"

function download_data()
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = read("$envdir/database/retrieval.txt", String)
    files = vcat(
        ["$m.groups.csv" for m in MEDIUMS],
        ["$(s)_$(m).csv" for s in SOURCES for m in MEDIUMS],
        ["fingerprints.csv"],
    )
    for fn in files
        cmd = "$retrieval/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
    run(
        `mlr --csv split -n 1000000 --prefix $datadir/fingerprints $datadir/fingerprints.csv`,
    )
    rm("$datadir/fingerprints.csv")
end

@memoize function get_media_length(source, medium)
    fn = "$datadir/$(source)_$(medium).csv"
    df = CSV.read(fn, DataFrames.DataFrame)
    rm(fn)
    parseint(x::Missing) = nothing
    parseint(x::Real) = x
    parseint(x::AbstractString) = parse(Int, replace(x, "+" => ""))
    Dict(
        string(uid) => Dict(
            "episodes" => parseint(eps),
            "chapters" => parseint(chs),
            "volumes" => parseint(vls),
        ) for (uid, eps, chs, vls) in zip(df.itemid, df.episodes, df.chapters, df.volumes)
    )
end

function shuffle_col!(df, col)
    defaultval = 0
    vals = Set(df[:, col])
    delete!(vals, defaultval)
    remap = Random.shuffle(collect(vals))
    remap = Dict(x => i for (i, x) in Iterators.enumerate(remap))
    remap[defaultval] = defaultval
    df[:, col] .= map(x -> remap[x], df[:, col])
end

@memoize function get_media_groups(medium)
    fn = "$datadir/$medium.groups.csv"
    df = CSV.read(
        fn,
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
    )
    rm(fn)
    cols = [:episodes, :chapters, :volumes]
    for c in cols
        df[!, c] .= 0
    end
    for i = 1:DataFrames.nrow(df)
        lens = get(get_media_length(df.source[i], medium), df.itemid[i], Dict())
        for c in cols
            df[i, c] = something(get(lens, string(c), nothing), 0)
        end
    end
    sort!(df, :count, rev = true)
    df[!, :distinctid] .= 0
    df[!, :matchedid] .= 0
    min_count = 100
    distinctid = 0
    groupmap = Dict()
    for i = 1:DataFrames.nrow(df)
        if df.count[i] < min_count
            df[i, :distinctid] = 0
            df[i, :matchedid] = get(groupmap, df[i, :groupid], 0)
        else
            distinctid += 1
            if df[i, :groupid] ∉ keys(groupmap)
                groupmap[df[i, :groupid]] = length(groupmap) + 1
            end
            df[i, :distinctid] = distinctid
            df[i, :matchedid] = groupmap[df[i, :groupid]]
        end
    end
    for c in [:distinctid, :matchedid]
        shuffle_col!(df, c)
    end
    df
end

function gen_splits()
    min_items = 5
    test_perc = 0.01
    rm("$datadir/users", recursive = true, force = true)
    @showprogress for (idx, f) in
                      Iterators.enumerate(Glob.glob("$datadir/fingerprints_*.csv"))
        train_dir = "$datadir/users/training/$idx"
        test_dir = "$datadir/users/test/$idx"
        mkpath.([train_dir, test_dir])
        df = Random.shuffle(read_csv(f))
        Threads.@threads for i = 1:DataFrames.nrow(df)
            user = import_user(df.source[i], decompress(df.data[i]))
            if length(user["items"]) < min_items
                continue
            end
            outdir = rand() < test_perc ? test_dir : train_dir
            open("$outdir/$i.msgpack", "w") do g
                write(g, MsgPack.pack(user))
            end
        end
        rm(f)
    end
end

function import_data()
    download_data()
    for m in MEDIUMS
        CSV.write("$datadir/$m.csv", get_media_groups(m))
    end
    date = Dates.format(Dates.today(), "yyyymmdd")
    open("$datadir/latest", "w") do f
        write(f, date)
    end
    save_template = read("$envdir/database/storage.txt", String)
    for m in MEDIUMS
        cmd = replace(
            save_template,
            "{INPUT}" => "$datadir/$m.csv",
            "{OUTPUT}" => "$date/$m.csv",
        )
        run(`sh -c $cmd`)
    end
    gen_splits()
end

import_data()
