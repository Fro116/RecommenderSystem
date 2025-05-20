import CSV
import DataFrames
import Dates
import Glob
import HTTP
import Memoize: @memoize
import MsgPack
import Random
import ProgressMeter: @showprogress
include("../julia_utils/stdout.jl")
include("../Training/import_list.jl")

const SOURCES = ["mal", "anilist", "kitsu", "animeplanet"]
const datadir = "../../data/finetune"

function blue_green_tag()
    curtag = nothing
    for x in ["blue", "green"]
        url = read("../../secrets/url.embed.$x.txt", String)
        r = HTTP.get("$url/ready", status_exception = false)
        if !HTTP.iserror(r)
            curtag = x
            break
        end
    end
    if isnothing(curtag)
        curtag = "blue"
    end
    Dict("blue" => "green", "green" => "blue")[curtag]
end

function download_data(finetune_tag::AbstractString)
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    download = "rclone --retries=10 copyto r2:rsys/database"
    cmd = "$download/training/latest $datadir/training_tag"
    run(`sh -c $cmd`)
    tag = read("$datadir/training_tag", String)
    files = vcat(
        ["$m.csv" for m in ["manga", "anime"]],
        ["media_relations.$m.jld2" for m in [0, 1]],
        ["transformer.$modeltype.$stem" for modeltype in ["retrieval", "ranking"] for stem in ["csv", "pt"]],
        ["images.csv"],
    )
    for fn in files
        cmd = "$download/training/$tag/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
    open("$datadir/finetune_tag", "w") do f
        write(f, finetune_tag)
    end
    tag = read("$datadir/finetune_tag", String)
    run(`rclone --retries=10 copyto r2:rsys/database/lists/$tag/histories.csv.zstd $datadir/histories.csv.zstd`)
    run(`unzstd $datadir/histories.csv.zstd`)
    rm("$datadir/histories.csv.zstd")
    run(
        `mlr --csv split -n 1000000 --prefix $datadir/histories $datadir/histories.csv`,
    )
    rm("$datadir/histories.csv")
    open("$datadir/bluegreen", "w") do f
        write(f, blue_green_tag())
    end
end

function num_items(m::AbstractString, source::AbstractString)
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1)
    filter!(x -> x.source == source, df)
    n = length(Set(df.matchedid))
    logtag("IMPORT_DATA", "num_items($m, $source) = $n")
    n
end

function gen_splits()
    mediums = ["manga", "anime"]
    recent_items = 5 # TODO test more items
    min_items = 5
    max_items = Dict(s => num_items.(mediums, s) for s in SOURCES)
    test_perc = 0.1
    training_recent_days = 1 # TODO test more days
    test_recent_days = 1
    training_tag = read("$datadir/training_tag", String)
    max_ts = -Inf
    @showprogress for f in Glob.glob("$datadir/histories_*.csv")
        df = read_csv(f)
        max_df_ts = maximum(parse.(Float64, df.db_refreshed_at))
        max_ts = max(max_ts, max_df_ts)
    end
    logtag("IMPORT_DATA", "using max_ts of $max_ts")
    rm("$datadir/users", recursive = true, force = true)
    @showprogress for (idx, f) in
                      Iterators.enumerate(Glob.glob("$datadir/histories_*.csv"))
        train_dir = "$datadir/users/training/$idx"
        test_dir = "$datadir/users/test/$idx"
        unused_dir = "$datadir/users/unused/$idx"
        mkpath.([train_dir, test_dir, unused_dir])
        df = Random.shuffle(read_csv(f))
        Threads.@threads for i = 1:DataFrames.nrow(df)
            access_ts = parse(Float64, df.db_refreshed_at[i])
            if access_ts < max_ts - max(training_recent_days, test_recent_days) * 86400
                continue
            end
            user = import_user(df.source[i], decompress(df.data[i]), access_ts)
            n_predict = 0
            n_items = zeros(Int, length(mediums))
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
                outdir = unused_dir
            elseif rand() < test_perc
                recent_days = test_recent_days
                outdir = test_dir
            else
                recent_days = training_recent_days
                outdir = train_dir
            end
            if outdir == unused_dir
                continue
            end
            recent_ts = max_ts - 86400 * recent_days
            old_items = []
            new_items = []
            for x in reverse(user["items"])
                if x["history_max_ts"] > recent_ts && length(new_items) < recent_items
                    if x["history_tag"] <= training_tag
                        continue
                    end
                    if isnothing(x["history_min_ts"]) || (x["history_max_ts"] - x["history_min_ts"] > 86400)
                        continue
                    end
                    if (x["status"] == x["history_status"]) && (x["rating"] == x["history_rating"])
                        continue
                    end
                    push!(new_items, x)
                else
                    push!(old_items, x)
                end
            end
            if isempty(new_items)
                continue
            end
            user["items"] = reverse(old_items)
            user["test_items"] = reverse(new_items)
            open("$outdir/$i.msgpack", "w") do g
                write(g, MsgPack.pack(user))
            end
        end
        rm(f)
    end
end

function count_test_items(datasplit)
    n_watch = [0, 0]
    n_rating = [0, 0]
    n_status = [0, 0]
    planned_status = 5
    for fn in Glob.glob("../../data/finetune/users/$datasplit/*/*.msgpack")
        data = open(fn) do f
            MsgPack.unpack(read(f))
        end
        for x in data["test_items"]
            m = x["medium"]
            inferred_watch = x["status"] == 0 && isnothing(x["history_status"])
            new_watch =
                (x["status"] > planned_status) &&
                (isnothing(x["history_status"]) || x["history_status"] <= planned_status)
            if inferred_watch || new_watch
                n_watch[m+1] += 1
            end
            if (x["rating"] > 0) && (x["rating"] != x["history_rating"])
                n_rating[m+1] += 1
            end
            if (x["status"] > 0) && (x["status"] != x["history_status"])
                n_status[m+1] += 1
            end
        end
    end
    logtag("TRANSFORMER", "$datasplit has $n_watch watch, $n_rating rating, and $n_status status entries")
end

download_data(ARGS[1])
gen_splits()
count_test_items.(["training", "test"])
