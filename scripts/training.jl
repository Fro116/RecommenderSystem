import CSV
import Dates
import DataFrames
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")

function days_since_last_train()
    str = read(`rclone lsf r2:rsys/database/training`, String)
    tags = sort([chop(x) for x in split(str) if endswith(x, "/")])
    if isempty(tags)
        return Inf
    end
    last_train = Dates.DateTime(tags[end], Dates.dateformat"yyyymmdd")
    (Dates.now() - last_train) / Dates.Day(1)
end

function runcmd(x)
    logtag("TRAINING", "running $x")
    run(`sh -c $x`)
end

function import_db(name::String)
    logdir = "../logs/import"
    if !ispath(logdir)
        mkpath(logdir)
    end
    teecmd(x, filename) = "($x) 2>&1 | tee $filename"
    runcmd(teecmd("cd Import/$name && julia save_$(name).jl", "$logdir/$name.log"))
end

function run_training()
    days = days_since_last_train()
    if days < 15
        return
    end
    logtag("TRAINING", "retraining after $days days")
    for x in ["media", "images", "embeddings", "autocomplete_users"]
        import_db(x)
    end
    datetag = Dates.format(Dates.today(), "yyyymmdd")
    latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    while datetag != latest
        logtag("TRAIN_MODELS", "waiting for $datetag, latest $latest")
        sleep(600)
        datetag = Dates.format(Dates.today(), "yyyymmdd")
        latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    end
    runcmd("cd Training && julia run.jl $latest")
    for x in ["autocomplete_items"]
        import_db(x)
    end
end

@scheduled "RUN_TRAINING" "11:00" @handle_errors run_training()
