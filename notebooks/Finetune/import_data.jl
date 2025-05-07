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
        ["baseline.$metric.$m.msgpack" for metric in ["rating"] for m in [0, 1]],
        ["bagofwords.$m.$metric.$stem" for m in [0, 1] for metric in ["rating"] for stem in ["csv", "pt"]],
        ["transformer.$stem" for stem in ["csv", "pt", "config"]],
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

function project!(user)
    # remove acausal deleted items
    max_history_tag = ""
    for x in user["items"]
        tag = x["history_tag"]
        if tag in ["infer", "delete"]
            continue
        end
        max_history_tag = max(max_history_tag, tag)
    end
    items = []
    for x in user["items"]
        if x["history_tag"] == max_history_tag && x["history_tag"] == "delete"
            continue
        end
        push!(items, x)
    end
    # set metrics for custom tags
    deleted_status = 3
    for x in items
        tag = x["history_tag"]
        if tag == "infer"
            x["status"] = 0
            x["rating"] = 0
            x["progress"] = 0
        elseif tag == "delete"
            x["status"] = deleted_status
            x["rating"] = 0
        end
    end
    # don't predict watched items
    prev_snapshot = Dict()
    for x in items
        if x["history_tag"] == max_history_tag
            continue
        end
        k = (x["medium"], x["matchedid"])
        if x["history_tag"] == "delete"
            delete!(prev_snapshot, k)
        else
            prev_snapshot[k] = x["status"]
        end
    end
    planned_status = 5
    for x in items
        predict = true
        k = (x["medium"], x["matchedid"])
        if k in keys(prev_snapshot) && prev_snapshot[k] != planned_status
            predict = false
        end
        x["history_predict"] = predict
    end
    user["items"] = items
end


function gen_splits()
    mediums = ["manga", "anime"]
    recent_items = 5 # TODO test more items
    min_items = 5
    max_items = Dict(s => num_items.(mediums, s) for s in SOURCES) # TODO study clip threshold
    test_perc = 0.1 # TODO test smaller test_perc
    max_ts = -Inf
    @showprogress for f in Glob.glob("$datadir/histories_*.csv")
        df = read_csv(f)
        max_df_ts = maximum(parse.(Float64, df.db_refreshed_at))
        max_ts = max(max_ts, max_df_ts)
    end
    # TODO think about timezone skew between db_refreshed_at and updated_at
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
            user = import_user(df.source[i], decompress(df.data[i]), parse(Float64, df.db_refreshed_at[i]))
            n_items = zeros(Int, length(mediums))
            for x in user["items"]
                n_items[x["medium"]+1] += 1
            end
            if sum(n_items) < min_items
                recent_days = 0
                outdir = unused_dir
            elseif any(n_items .> max_items[df.source[i]])
                k = (df.source[i], df.username[i], df.userid[i])
                logerror("skipping user $i: $k with $n_items > $(max_items[df.source[i]]) items")
                recent_days = 0
                outdir = unused_dir
            elseif rand() < test_perc
                recent_days = 1
                outdir = test_dir
            else
                recent_days = 1 # TODO test more days
                outdir = train_dir
            end
            project!(user)
            recent_ts = max_ts - 86400 * recent_days
            old_items = []
            new_items = []
            for x in reverse(user["items"])
                if x["history_max_ts"] > recent_ts && length(new_items) < recent_items
                    if !x["history_predict"]
                        continue
                    end
                    push!(new_items, x)
                else
                    push!(old_items, x)
                end
            end
            if isempty(new_items)
                outdir = unused_dir
            end
            if outdir == unused_dir
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
