import Dates
include("julia_utils/multithreading.jl")
include("julia_utils/scheduling.jl")
include("julia_utils/stdout.jl")

function runcmd(x)
    logtag("CRON", "running $x")
    run(`sh -c $x`)
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

function import_dbs()
    for x in ["media", "images", "autocomplete", "autocomplete_items"]
        runcmd("cd Import/$x && julia save_$(x).jl")
    end
end

function train_models()
    today = Dates.today()
    if Dates.dayofweek(today) == 3
        import_dbs()
    end
    if Dates.day(today) in [15]
        runcmd("cd Training && julia run.jl")
    end
    datetag = Dates.format(today, "yyyymmdd")
    latest = read(`rclone cat r2:rsys/database/lists/latest`, String)
    if datetag != latest
        logerror("TRAIN_MODELS", "list $datetag is not ready, using $latest")
    end
    runcmd("cd Finetune && julia run.jl $latest")
end

@scheduled "IMPORT_LISTS" "2:30" @handle_errors import_lists()
@scheduled "TRAIN_MODELS" "9:00" @handle_errors train_models()
