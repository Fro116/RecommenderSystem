import Dates
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")

const gpulock = ReentrantLock()

function runcmd(x)
    logtag("CRON", "running $x")
    run(`sh -c $x`)
end

function teecmd(x, filename)
    "($x) 2>&1 | tee $filename"
end

function import_db(name::String)
    logdir = "../logs/import"
    if !ispath(logdir)
        mkpath(logdir)
    end
    runcmd(teecmd("cd Import/$name && julia save_$(name).jl", "$logdir/$name.log"))
end

function import_dbs()
    if Dates.dayofweek(Dates.today()) != Dates.Monday
        return
    end
    if Dates.dayofmonth(Dates.today()) <= 7
        import_db("embeddings")
    end
    lock(gpulock) do
        import_db("images")
    end
    for x in ["media", "autocomplete", "autocomplete_items"]
        import_db(x)
    end
end

function run_training()
    if Dates.dayofmonth(Dates.today()) ∉ [8, 23]
        return
    end
    lock(gpulock) do
        datetag = Dates.format(Dates.today(), "yyyymmdd")
        latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
        if datetag != latest
            logtag("TRAIN_MODELS", "list $datetag is not ready, using $latest")
        end
        runcmd("cd Training && julia run.jl $latest")
    end
end

function run_finetune()
    lock(gpulock) do
        datetag = Dates.format(Dates.today(), "yyyymmdd")
        latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
        if datetag != latest
            logtag("TRAIN_MODELS", "list $datetag is not ready, using $latest")
        end
        runcmd("cd Finetune && julia run.jl $latest")
    end
end

Threads.@spawn @scheduled "IMPORT_DBS" "01:00" @handle_errors import_dbs()
Threads.@spawn @scheduled "RUN_TRAINING" "09:59" @handle_errors run_training()
@scheduled "RUN_FINETUNE" "10:00" @handle_errors run_finetune()
