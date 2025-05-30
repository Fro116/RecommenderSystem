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
    for x in ["autocomplete", "images", "media"]
        runcmd("cd Import/$x && julia save_$(x).jl")
    end
end

function cron()
    today = Dates.today()
    import_lists()
    if Dates.dayofweek(today) == 1
        import_dbs()
    end
    # for now, manually oversee training runs
    # if Dates.day(today) in [1, 15]
    #     runcmd("cd Training && julia run.jl")
    # end
    datetag = Dates.format(today, "yyyymmdd")
    runcmd("cd Finetune && julia run.jl $datetag")
end

@scheduled "CRON" "2:30" @handle_errors cron()
