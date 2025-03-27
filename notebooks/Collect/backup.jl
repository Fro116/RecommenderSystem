include("../julia_utils/database.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function backup_dbs()
    logtag("BACKUP", "DATABASES")
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
        "external_dependencies",
    ]
    save_template = "> cloudstorage.tmp; PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin rclone --retries=10 copyto cloudstorage.tmp r2:rsys/database/collect/{FILE}; rm cloudstorage.tmp"
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
    cleanup = raw"rclone lsd r2:rsys/database/collect/ | sort | head -n -30 | awk '{print $NF}' | xargs -I {} rclone purge r2:rsys/database/collect/{}"
    run(`sh -c $cleanup`)
    cmd = "cd ../Import/lists && julia save_lists.jl"
    run(`sh -c $cmd`)
end

function backup_images()
    logtag("BACKUP", "IMAGES")
    datadir = "../../data/collect"
    run(`rclone --retries=10 sync $datadir/images r2:cdn/images/cards --exclude '*~'`)
    run(`rclone --retries=10 copyto $datadir/images.csv r2:rsys/database/import/images.csv`)
end

function backup()
    backup_dbs()
    backup_images()
end

@scheduled "BACKUP" "01:00" @handle_errors backup()
