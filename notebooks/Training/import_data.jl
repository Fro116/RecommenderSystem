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

function download_data()
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = "rclone --retries=10 copyto r2:rsys/database/import"
    files = vcat(
        ["$m.groups.csv" for m in MEDIUMS],
        ["$(s)_$(m).csv" for s in SOURCES for m in MEDIUMS],
        ["$(s)_media_relations.csv" for s in SOURCES],
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

function get_media(source, medium::String)
    fn = "$datadir/$(source)_$(medium).csv"
    df = CSV.read(fn, DataFrames.DataFrame)
    parseint(x::Missing) = missing
    parseint(x::Real) = x
    parseint(x::AbstractString) = parse(Int, replace(x, "+" => ""))
    for c in [:episodes, :chapters, :volumes]
        df[!, c] = parseint.(df[:, c])
    end
    df[!, :source_material] = df[:, :source]
    df[!, :source] = fill(source, DataFrames.nrow(df))
    medium_map = Dict("manga" => 0, "anime" => 1)
    df[!, :medium] = fill(medium_map[medium], DataFrames.nrow(df))
    df[!, :itemid] = string.(df[:, :itemid])
    df = df[:, DataFrames.Not([:malid, :anilistid])]
    df
end

get_media(medium::String) = reduce(vcat, [get_media(s, medium) for s in SOURCES])

function shuffle_col!(df, col)
    defaultval = 0
    vals = Set(df[:, col])
    delete!(vals, defaultval)
    remap = Random.shuffle(collect(vals))
    remap = Dict(x => i for (i, x) in Iterators.enumerate(remap))
    remap[defaultval] = defaultval
    df[:, col] .= map(x -> remap[x], df[:, col])
end

function get_media_groups(medium::AbstractString)
    fn = "$datadir/$medium.groups.csv"
    groups = CSV.read(fn, DataFrames.DataFrame, types = Dict("itemid" => String))
    media = get_media(medium)
    df = DataFrames.innerjoin(groups, media, on = [:source, :itemid])
    for c in [:episodes, :chapters, :volumes]
        df[!, c] = coalesce.(df[:, c], 0)
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

function get_idmaps()
    media = Dict(
        k => CSV.read("$datadir/$m.csv", DataFrames.DataFrame) for
        (k, m) in [(0, "manga"), (1, "anime")]
    )
    matched2source = Dict()
    source2matched = Dict()
    for m in [0, 1]
        df = media[m]
        for i = 1:DataFrames.nrow(df)
            if df.matchedid[i] == 0
                continue
            end
            mkey = (m, df.matchedid[i])
            skey = (m, df.source[i], string(df.itemid[i]))
            if mkey ∉ keys(matched2source)
                matched2source[mkey] = skey
            end
            source2matched[skey] = mkey
        end
    end
    Dict("matched" => matched2source, "source" => source2matched)
end

function get_media_details()
    d = Dict()
    for medium in ["manga", "anime"]
        df = CSV.read("$datadir/$medium.csv", DataFrames.DataFrame)
        for i = 1:DataFrames.nrow(df)
            k = (df.medium[i], df.matchedid[i])
            d[k] = Dict("mediatype" => df.mediatype[i])
        end
    end
    d
end

function get_relations()
    idmaps = get_idmaps()
    medium_map = Dict("manga" => 0, "anime" => 1)
    relations = []
    for s in ["mal", "anilist", "kitsu", "animeplanet"]
        df = CSV.read("$datadir/$(s)_media_relations.csv", DataFrames.DataFrame)
        for i = 1:DataFrames.nrow(df)
            skey = (medium_map[df.medium[i]], s, string(df.itemid[i]))
            tkey = (medium_map[df.target_medium[i]], s, string(df.target_id[i]))
            mskey = get(idmaps["source"], skey, nothing)
            mtkey = get(idmaps["source"], tkey, nothing)
            if isnothing(mskey) || isnothing(mtkey)
                continue
            end
            if idmaps["matched"][mskey] != skey
                continue
            end
            push!(relations, (mskey..., mtkey..., df.relation[i]))
        end
    end
    details = get_media_details()
    manga_types = Set(["Manhwa", "Manhua", "Manga", "OEL", "Doujinshi", "One-shot"])
    novel_types = Set(["Light Novel", "Novel"])
    for i = 1:length(relations)
        m1, id1, m2, id2, r = relations[i]
        if m1 != m2 && r ∉ ["adaptation", "source"]
            relations[i] = (m1, id1, m2, id2, "adaptation")
            continue
        end
        if r == "unknown"
            d1 = details[(m1, id1)]["mediatype"]
            d2 = details[(m2, id2)]["mediatype"]
            if d1 in manga_types && d2 in novel_types ||
               d1 in novel_types && d2 in manga_types
                relations[i] = (m1, id1, m2, id2, "adaptation")
                continue
            end
        end
    end
    DataFrames.DataFrame(
        relations,
        [:source_medium, :source_matchedid, :target_medium, :target_matchedid, :relation],
    )
end

function gen_splits()
    min_items = 5
    test_perc = 0.01
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
    CSV.write("$datadir/media_relations.csv", get_relations())
    date = Dates.format(Dates.today(), "yyyymmdd")
    open("$datadir/latest", "w") do f
        write(f, date)
    end
    save_template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/training/{OUTPUT}"
    files = ["$m.csv" for m in MEDIUMS]
    for f in files
        cmd = replace(
            save_template,
            "{INPUT}" => "$datadir/$f",
            "{OUTPUT}" => "$date/$f",
        )
        run(`sh -c $cmd`)
    end
    gen_splits()
end

import_data()
