import CSV
import Dates
import DataFrames
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")

const gpulock = ReentrantLock()

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
    if Dates.dayofmonth(Dates.today()) ∉ [8, 23]
        return
    end
    for x in ["media", "autocomplete_users", "autocomplete_items", "embeddings"]
        import_db(x)
    end
    lock(gpulock) do
        import_db("images")
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


@scheduled "RUN_TRAINING" "07:00" @handle_errors run_training()
