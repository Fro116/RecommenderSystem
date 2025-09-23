import Dates
include("julia_utils/multithreading.jl")
include("julia_utils/scheduling.jl")
include("julia_utils/stdout.jl")

function runcmd(x)
    logtag("CRON", "running $x")
    run(`sh -c $x`)
end

function teecmd(x, filename)
    "($x) 2>&1 | tee $filename"
end

function get_directories(db)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    dirs = [chop(x) for x in split(str) if endswith(x, "/")]
    latest = read(`rclone cat r2:rsys/database/$db/latest`, String)
    [x for x in dirs if x <= latest]
end

function import_lists()
    tags_to_import = sort(collect(setdiff(Set.(get_directories.(["collect", "lists"]))...)))
    for tag in tags_to_import
        runcmd("cd Import/lists && julia save.jl $tag")
    end
end

function import_db(name::String)
    logdir = "../logs/import"
    if !ispath(logdir)
        mkpath(logdir)
    end
    runcmd(teecmd("cd Import/$name && julia save_$(name).jl", "$logdir/$name.log"))
end

function import_dbs()
    dbs = Dict(
        1 => ["media"],
        2 => ["images"], # 9h
        3 => ["autocomplete"],
        4 => ["autocomplete_items"],
    )
    dow = Dates.dayofweek(Dates.today())
    if dow ∉ keys(dbs)
        return
    end
    import_db.(dbs[dow])
end

function run_training()
    import_db("embeddings")
    datetag = Dates.format(Dates.today(), "yyyymmdd")
    latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    if datetag != latest
        logtag("TRAIN_MODELS", "list $datetag is not ready, using $latest")
    end
    runcmd("cd Training && julia run.jl $latest")
end

function run_finetune()
    datetag = Dates.format(Dates.today(), "yyyymmdd")
    latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    if datetag != latest
        logtag("TRAIN_MODELS", "list $datetag is not ready, using $latest")
    end
    runcmd("cd Finetune && julia run.jl $latest")
end

function train_models()
    # if Dates.day(Dates.today()) == 15
    #     run_training()
    # end
    import_dbs()
    run_finetune()
end

Threads.@spawn @scheduled "IMPORT_LISTS" "2:30" @handle_errors import_lists()
Threads.@spawn @scheduled "TRAIN_MODELS" "10:00" @handle_errors train_models()
while true
    sleep(86400)
end