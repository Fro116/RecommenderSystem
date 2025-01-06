include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function backup()
    tables = [
        "mal_userids",
        "mal_users",
        "mal_user_items",
        "mal_media",
        "mal_media_relations",
        "anilist_users",
        "anilist_user_items",
        "anilist_media",
        "anilist_media_relations",
        "kitsu_users",
        "kitsu_user_items",
        "kitsu_media",
        "kitsu_media_relations",
        "animeplanet_userids",
        "animeplanet_users",
        "animeplanet_user_items",
        "animeplanet_media",
        "animeplanet_media_relations",
    ]
    save_template = read("$DB_PATH/storage.txt", String)
    date = Dates.format(Dates.today(), "yyyymmdd")
    with_db(:garbage_collect) do db
        for table in tables
            logtag("BACKUP", table)
            save = replace(save_template, "{FILE}" => "$date/$table.zstd")
            query = "COPY $table TO PROGRAM 'zstd $save' WITH (FORMAT CSV, HEADER, FORCE_QUOTE *);"
            stmt = db_prepare(db, query)
            LibPQ.execute(stmt)
        end
    end
    save = replace(save_template, "{FILE}" => "latest")
    cmd = "echo -n $date $save"
    run(`sh -c $cmd`)
    logtag("BACKUP", "pg_dump")
    dump = read("$DB_PATH/dump.txt", String)
    save = replace(save_template, "{FILE}" => "database.sql.zst")
    cmd = "$dump $save"
    run(`sh -c $cmd`)
end

@scheduled "BACKUP" "04:00" @handle_errors backup()
