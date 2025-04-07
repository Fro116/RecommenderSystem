include("julia_utils/multithreading.jl")
include("julia_utils/scheduling.jl")
include("julia_utils/stdout.jl")

function runcmd(x)
    logtag("CRON", "running $x")
    run(`sh -c $x`)
end

function cron()
    day = Dates.day(Dates.today())
    for x ["lists", "autocomplete", "images", "media"]
        runcmd("cd Import/$x && julia save_$(x).jl")
    end
    train_models = false
    if train_models
        if day in [1, 8, 15, 22]
            runcmd("cd Training && julia run.jl")
        end
        runcmd("cd Finetune && julia run.jl")
    end
end

@scheduled "CRON" "2:00" @handle_errors cron()
