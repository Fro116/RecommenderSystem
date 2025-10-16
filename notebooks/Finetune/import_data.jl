import CSV
import DataFrames
import Dates
import Glob
import HTTP
import Memoize: @memoize
import MsgPack
import Random
import ProgressMeter: @showprogress
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../Training/import_list.jl")
include("../Training/history_tools.jl")

const SOURCES = ["mal", "anilist", "kitsu", "animeplanet"]
const datadir = "../../data/finetune"

function download_data(finetune_tag::AbstractString)
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    download = "rclone --retries=10 copyto r2:rsys/database"
    cmd = "$download/training/latest $datadir/training_tag"
    run(`sh -c $cmd`)
    tag = read("$datadir/training_tag", String)
    files = vcat(
        ["$m.csv" for m in ["manga", "anime"]],
        ["$m.json" for m in ["manga", "anime"]],
        ["media_relations.$m.jld2" for m in [0, 1]],
        ["watches.$m.jld2" for m in [0, 1]],
        ["transformer.$modeltype.$stem" for modeltype in ["causal", "masked"] for stem in ["csv", "pt"]],
        ["pairwise.embeddings.$stem" for stem in ["jld2", "csv"]],
        ["images.csv", "media_relations.csv"],
        ["media_embeddings.h5" ],
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
    recent_items = 1
    min_items = 5
    max_items = Dict(s => num_items.(mediums, s) for s in SOURCES)
    test_perc = 0.1
    training_recent_days = 365/4
    test_recent_days = 365/4
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
        mkpath.([train_dir, test_dir])
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
                outdir = nothing
            elseif any(n_items .> max_items[df.source[i]])
                k = (df.source[i], df.username[i], df.userid[i])
                outdir = nothing
            elseif rand() < test_perc
                recent_days = test_recent_days
                outdir = test_dir
            else
                recent_days = training_recent_days
                outdir = train_dir
            end
            if isnothing(outdir)
                continue
            end
            tokenize!(user)
            recent_ts = max_ts - 86400 * recent_days
            old_items = []
            new_items = []
            for x in reverse(user["items"])
                if x["history_max_ts"] > recent_ts && length(new_items) < recent_items
                    if x["history_tag"] in ["infer", "delete"]
                        continue # filter to recorded events
                    end
                    if x["history_tag"] <= training_tag
                        continue # filter to items that are out-of-sample from pretraining
                    end
                    if isnothing(x["history_min_ts"]) || (x["history_max_ts"] - x["history_min_ts"] > 86400)
                        continue # filter to items with reliable timestamps
                    end
                    if (x["status"] == x["history_status"]) && (x["rating"] == x["history_rating"])
                        continue # filter to non-trivial updates
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

download_data(ARGS[1])
gen_splits()
