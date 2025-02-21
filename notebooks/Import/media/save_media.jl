include("../../julia_utils/multithreading.jl")
include("../../julia_utils/scheduling.jl")
include("../../julia_utils/stdout.jl")
include("common.jl")

function backup()
    open("../../../data/match_media.log", "w") do f
        for fn in [
            "download_media.jl",
            "match_ids.jl",
            "match_manami.jl",
            "match_manual.jl",
            "match_metadata.jl",
            "match_media.jl",
        ]
            run(pipeline(`julia $fn`, stdout=f, stderr=f))
        end
    end
    save_template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/import/{OUTPUT}"
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

@scheduled "BACKUP" "22:00" @handle_errors backup()
