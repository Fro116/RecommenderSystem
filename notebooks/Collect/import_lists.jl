include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function runcmd(x)
    logtag("IMPORT_LISTS", "running $x")
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
        runcmd("cd ../Import/lists && julia save.jl $tag")
    end
end

import_lists()
@scheduled "IMPORT_LISTS" "2:30" @handle_errors import_lists()
