include("database.jl")
include("http.jl")

function backup(secs::Int)
    while true
        curtime = time()
        logtag("BACKUP", "START")
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
        path = read("$DB_PATH/storage.txt", String)
        date = Dates.format(Dates.today(), "yyyymmdd")
        with_db(:garbage_collect) do db
            for table in tables
                logtag("BACKUP", table)
                query = "COPY $table TO PROGRAM 'zstd | $path/$date/$table.zstd' WITH (FORMAT CSV, HEADER);"
                stmt = db_prepare(db, query)
                LibPQ.execute(stmt)
            end
        end
        logtag("BACKUP", "pg_dump")
        dump = read("$DB_PATH/dump.txt", String)
        cmd = "$dump | $path/database.sql.zst"
        run(`sh -c $cmd`)
        logtag("BACKUP", "END")
        sleep_secs = secs - (time() - curtime)
        if sleep_secs < 0
            logtag("BACKUP", "late by $sleep_secs seconds")
        else
            sleep(sleep_secs)
        end
    end
end

sleep(3600) # run off-cycle
backup(86400)
