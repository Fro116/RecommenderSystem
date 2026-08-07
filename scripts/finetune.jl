import CSV
import Dates
import DataFrames
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")

function runcmd(x)
    logtag("FINETUNE", "running $x")
    run(`sh -c $x`)
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
    latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    if latest == last_date
        return
    end
    runcmd("cd Finetune && julia run.jl $latest")
    last_date = latest
end

@periodic "RUN_FINETUNE" 600 @handle_errors run_finetune()
