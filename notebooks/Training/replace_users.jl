import CSV
import DataFrames
import ProgressMeter: @showprogress

include("../julia_utils/http.jl")
include("../julia_utils/database.jl")
const envdir = "../../environment"
const datadir = "../../data/replace_users"

function download_users()
    mkpath(datadir)
    retrieval = read("$envdir/database/retrieval.txt", String)
    cmd = "$retrieval/latest $datadir/latest"
    run(`sh -c $cmd`)
    tag = read("$datadir/latest", String)
    rm("$datadir/latest")
    for s in ["mal", "anilist", "kitsu", "animeplanet"]
        cmd = "$retrieval/$tag/$(s)_users.zstd $datadir/$(s)_users.csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $datadir/$(s)_users.csv.zstd`)
        rm("$datadir/$(s)_users.csv.zstd")
    end
end

function get_entries(source, name_col, manga_col, anime_col)
    df = CSV.read("$datadir/$(source)_users.csv", DataFrames.DataFrame)
    entries = []
    @showprogress for i = 1:DataFrames.nrow(df)
        vals = (source, df[i, name_col], df.userid[i], df[i, manga_col], df[i, anime_col])
        if any(ismissing.(vals))
            continue
        end
        push!(entries, vals)
    end
    entries
end

function save_users()
    entries = vcat(
        get_entries("mal", "username", "manga_count", "anime_count"),
        get_entries("anilist", "username", "mangacount", "animecount"),
        get_entries("kitsu", "name", "manga_count", "anime_count"),
        get_entries("animeplanet", "username", "manga_count", "anime_count"),
    )
    ret = DataFrames.DataFrame(
        entries,
        [:source, :username, :userid, :manga_count, :anime_count],
    )
    CSV.write("$datadir/users.csv", ret)
end

function write_to_db()
    conn_str = read("$envdir/database/primary.txt", String)
    run(`psql $conn_str -f replace_users.sql`)
end

function cleanup()
    rm("$datadir/users.csv")
    for x in ["mal", "anilist", "kitsu", "animeplanet"]
        rm("$datadir/$(x)_users.csv")
    end
    rm(datadir)
end

download_users()
save_users()
write_to_db()
cleanup()
