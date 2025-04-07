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

function import_dbs()
    tags_to_import = sort(collect(setdiff(Set.(get_directories.(["collect", "lists"]))...)))
    for tag in tags_to_import
        runcmd("cd Import/lists && julia save_lists.jl $tag")
    end
    for x in ["autocomplete", "images", "media"]
        runcmd("cd Import/$x && julia save_$(x).jl")
    end
end

function cron()
    import_dbs()
    day = Dates.day(Dates.today())
    if day in [1, 8, 15, 22]
        runcmd("cd Training && julia run.jl")
    end
    runcmd("cd Finetune && julia run.jl")
end

@scheduled "CRON" "2:00" @handle_errors cron()
