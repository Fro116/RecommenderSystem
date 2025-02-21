import CSV
import DataFrames
import Dates
import Glob
import Memoize: @memoize
import MsgPack
import Random
import ProgressMeter: @showprogress
include("../julia_utils/stdout.jl")
include("../Training/import_list.jl")

const METRICS = ["rating", "watch", "plantowatch", "drop"]
const SOURCES = ["mal", "anilist", "kitsu", "animeplanet"]
const datadir = "../../data/finetune"

function download_data()
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    download = "rclone --retries=10 copyto r2:rsys/database"
    cmd = "$download/training/latest $datadir/training_tag"
    run(`sh -c $cmd`)
    tag = read("$datadir/training_tag", String)
    files = vcat(
        ["$m.csv" for m in ["manga", "anime"]],
        ["baseline.$m.msgpack" for m in [0, 1]],
        ["bagofwords.$m.$metric.pt" for m in [0, 1] for metric in METRICS],
    )
    for fn in files
        cmd = "$download/training/$tag/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
    cmd = "$download/import/fingerprints.csv $datadir/fingerprints.csv"
    run(`sh -c $cmd`)
    run(
        `mlr --csv split -n 1000000 --prefix $datadir/fingerprints $datadir/fingerprints.csv`,
    )
    rm("$datadir/fingerprints.csv")
end

function save_tag()
    date = Dates.format(Dates.today(), "yyyymmdd")
    open("$datadir/latest", "w") do f
        write(f, date)
    end
    save_template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/finetune/{OUTPUT}"
    cmd = replace(
        save_template,
        "{INPUT}" => "$datadir/latest",
        "{OUTPUT}" => "$date/latest",
    )
    run(`sh -c $cmd`)
    files = vcat(
        ["$m.csv" for m in ["manga", "anime"]],
        ["baseline.$m.msgpack" for m in [0, 1]],
        ["bagofwords.$m.$metric.pt" for m in [0, 1] for metric in METRICS],
    )
    for fn in files
        cmd = replace(
            save_template,
            "{INPUT}" => "$datadir/$fn",
            "{OUTPUT}" => "$date/$fn",
        )
        run(`sh -c $cmd`)
    end
end

function gen_splits()
    recent_days = 7
    recent_items = 5
    min_items = 5
    test_perc = 0.01
    max_ts = -Inf
    @showprogress for f in Glob.glob("$datadir/fingerprints_*.csv")
        df = read_csv(f)
        max_df_ts = maximum(parse.(Float64, filter(x -> !ismissing(x), df.db_refreshed_at)))
        max_ts = max(max_ts, max_df_ts)
    end
    recent_ts = max_ts - 86400 * recent_days
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
            old_items = []
            new_items = []
            for x in reverse(user["items"])
                if x["updated_at"] > recent_ts && length(new_items) < recent_items
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
            outdir = rand() < test_perc ? test_dir : train_dir
            open("$outdir/$i.msgpack", "w") do g
                write(g, MsgPack.pack(user))
            end
        end
        rm(f)
    end
end

download_data()
save_tag()
gen_splits()
