include("../../julia_utils/multithreading.jl")
include("../../julia_utils/scheduling.jl")
include("../../julia_utils/stdout.jl")
include("common.jl")
const logdir = "../../../data"

function backup()
    for fn in [
        "download_media.jl",
        "match_ids.jl",
        "match_manami.jl",
        "match_manual.jl",
        "match_metadata.jl",
        "match_media.jl",
    ]
        logfile = split(fn, ".")[1] * ".log"
        cmd = "julia $fn > $logdir/$logfile"
        run(`sh -c $cmd`)
    end
    save_template = read("$envdir/database/storage.txt", String)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    mediums = ["manga", "anime"]
    files = vcat(
        ["$(source)_$(medium).csv" for source in sources for medium in mediums],
        ["$medium.groups.csv" for medium in mediums]
    )
    for fn in files
        cmd = replace(save_template, "{INPUT}" => "$datadir/$fn", "{OUTPUT}" => fn)
        run(`sh -c $cmd`)
    end
end

@scheduled "BACKUP" "13:00" @handle_errors backup()
