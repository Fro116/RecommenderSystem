import CSV
import Dates
import DataFrames
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

function run_training()
    if Dates.dayofmonth(Dates.today()) ∉ [8, 23]
        return
    end
    for x in ["media", "autocomplete", "autocomplete_items", "embeddings"]
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

function get_last_finetune_date()
    text = read(`rclone cat r2:rsys/database/import/metrics.finetune.usermodel.csv`, String)
    if isempty(text)
        return nothing
    end
    df = CSV.read(IOBuffer(text), DataFrames.DataFrame)
    string(maximum(df.finetune_tag))
end

function run_finetune()
    last_date = get_last_finetune_date()
    while true
        lock(gpulock) do
            latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
            if latest == last_date
                return
            end
            runcmd("cd Finetune && julia run.jl $latest")
            last_date = latest
        end
        sleep(3600)
    end
end

Threads.@spawn @handle_errors run_finetune()
@scheduled "RUN_TRAINING" "02:00" @handle_errors run_training()
