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

const MEDIUMS = ["manga", "anime"]
const SOURCES = ["mal", "anilist", "kitsu", "animeplanet"]
const datadir = "../../data/finetune"
const envdir = "../../environment"

function download_data()
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = read("$envdir/database/retrieval.txt", String)
    files = vcat(
        ["$m.csv" for m in MEDIUMS],
        ["baseline.$m.csv" for m in MEDIUMS],
        ["bagofwords.$m.$metric.pt" for m in MEDIUMS for metric in METRICS],
    )
    for fn in files
        cmd = "$retrieval/$fn $mediadir/$fn"
        run(`sh -c $cmd`)
    end
    run(
        `mlr --csv split -n 1000000 --prefix $datadir/fingerprints $datadir/fingerprints.csv`,
    )
    rm("$datadir/fingerprints.csv")
end

function gen_splits()
    min_items = 5
    test_perc = 0.01
    recent_ts = Dates.datetime2unix(Dates.now())
    recent_items = 5
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
            for x in user["items"]
                if x["updated_at"] > recent_ts || !isempty(new_items)
                    push!(new_items, x)
                else
                    push!(old_items, x)
                end
                if isempty(new_items)
                    continue
                end
            end
            if isempty(new_items)
                continue
            end
            user["items"] = old_items
            user["test_items"] = new_items[1:recent_items]
            outdir = rand() < test_perc ? test_dir : train_dir
            open("$outdir/$i.msgpack", "w") do g
                write(g, MsgPack.pack(user))
            end
        end
        rm(f)
    end
end

download_data()
gen_splits()
