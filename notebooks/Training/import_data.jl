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

function download_data(datetag::AbstractString)
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    open("$datadir/list_tag", "w") do f
        write(f, datetag)
    end
    tag = read("$datadir/list_tag", String)
    run(
        `rclone --retries=10 copyto r2:rsys/database/lists/$tag/histories.csv.zstd $datadir/histories.csv.zstd`,
    )
    run(`unzstd $datadir/histories.csv.zstd`)
    run(`rm $datadir/histories.csv.zstd`)
    run(`mlr --csv split -n 1000000 --prefix $datadir/histories $datadir/histories.csv`)
    rm("$datadir/histories.csv")
    retrieval = "rclone --retries=10 copyto r2:rsys/database/import"
    files = vcat(
        ["$(s)_$(m).csv" for s in SOURCES for m in MEDIUMS],
        ["$m.groups.csv" for m in MEDIUMS],
        ["$(s)_media_relations.csv" for s in SOURCES],
        ["images.csv"],
        ["embeddings.json", "search_embeddings.jld2"],
    )
    for fn in files
        cmd = "$retrieval/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
end

function get_media(source, medium::String)
    fn = "$datadir/$(source)_$(medium).csv"
    df = CSV.read(fn, DataFrames.DataFrame, ntasks = 1)
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
    groups = copy(JSON3.read("$datadir/embeddings.json"))
    groups = [x for x in groups if x[:metadata][:medium] == medium]
    mids = Random.shuffle(1:length(groups))
    for i = 1:length(groups)
        groups[i][:matchedid] = mids[i]
    end
    open("$datadir/$medium.json", "w") do f
        JSON3.write(f, groups)
    end
    group_map = Dict()
    for x in groups
        for k in x[:keys]
            group_map[k] = x[:matchedid]
        end
    end
    counts = CSV.read("$datadir/$medium.groups.csv", DataFrames.DataFrame)
    counts = Dict((x.source, x.itemid) => x.count for x in eachrow(counts))
    df = get_media(medium)
    df[:, :count] = [get(counts, (x.source, x.itemid), 0) for x in eachrow(df)]
    for c in [:episodes, :chapters, :volumes]
        df[!, c] = coalesce.(df[:, c], 0)
    end
    df[!, :distinctid] .= 0
    df[!, :matchedid] .= 0
    distinctid = 0
    for i = 1:DataFrames.nrow(df)
        k = [medium, df.source[i], df.itemid[i]]
        df[i, :matchedid] = get(group_map, k, 0)
        if df[i, :matchedid] == 0
            df[i, :distinctid] = 0
        else
            distinctid += 1
            df[i, :distinctid] = distinctid
        end
    end
    shuffle_col!(df, :distinctid)
    df
end

function get_idmaps()
    media = Dict(
        k => CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1) for
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

function get_relations()
    idmaps = get_idmaps()
    medium_map = Dict("manga" => 0, "anime" => 1)
    relations = []
    for s in ["mal", "anilist", "kitsu", "animeplanet"]
        df = CSV.read("$datadir/$(s)_media_relations.csv", DataFrames.DataFrame, ntasks = 1)
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
    DataFrames.DataFrame(
        relations,
        [:source_medium, :source_matchedid, :target_medium, :target_matchedid, :relation],
    )
end

function num_items(m::AbstractString, source::AbstractString)
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1)
    filter!(x -> x.source == source, df)
    n = length(Set(df.matchedid))
    logtag("IMPORT_DATA", "num_items($m, $source) = $n")
    n
end

function gen_splits()
    min_items = 5
    max_items = Dict(s => num_items.(MEDIUMS, s) for s in SOURCES)
    test_perc = 0.01
    @showprogress for (idx, f) in Iterators.enumerate(Glob.glob("$datadir/histories_*.csv"))
        train_dir = "$datadir/users/training/$idx"
        test_dir = "$datadir/users/test/$idx"
        unused_dir = "$datadir/users/unused/$idx"
        mkpath.([train_dir, test_dir, unused_dir])
        df = Random.shuffle(read_csv(f))
        Threads.@threads for i = 1:DataFrames.nrow(df)
            user = import_user(
                df.source[i],
                decompress(df.data[i]),
                parse(Float64, df.db_refreshed_at[i]),
            )
            n_predict = 0
            n_items = zeros(Int, length(MEDIUMS))
            for x in user["items"]
                n_items[x["medium"]+1] += 1
                if x["status"] != x["history_status"] || x["rating"] != x["history_rating"]
                    n_predict += 1
                end
            end
            if n_predict < min_items
                outdir = unused_dir
            elseif any(n_items .> max_items[df.source[i]])
                k = (df.source[i], df.username[i], df.userid[i])
                logerror(
                    "skipping user $i: $k with $n_items > $(max_items[df.source[i]]) items",
                )
                outdir = unused_dir
            else
                outdir = rand() < test_perc ? test_dir : train_dir
            end
            if outdir == unused_dir
                continue
            end
            open("$outdir/$i.msgpack", "w") do g
                write(g, MsgPack.pack(user))
            end
        end
        rm(f)
    end
end

function import_data(datetag::AbstractString)
    download_data(datetag)
    for m in MEDIUMS
        CSV.write("$datadir/$m.csv", get_media_groups(m))
    end
    CSV.write("$datadir/media_relations.csv", get_relations())
    gen_splits()
    save_template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/training/{OUTPUT}"
    files = vcat(
        ["$m.csv" for m in MEDIUMS],
        ["list_tag", "images.csv", "media_relations.csv"],
        ["$m.json" for m in MEDIUMS],
    )
    for f in files
        cmd =
            replace(save_template, "{INPUT}" => "$datadir/$f", "{OUTPUT}" => "$datetag/$f")
        run(`sh -c $cmd`)
    end
end

import_data(ARGS[1])
