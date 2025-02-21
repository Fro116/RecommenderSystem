import CSV
import DataFrames
import JSON3
include("../../julia_utils/stdout.jl")
include("common.jl")

function download_media(source::String)
    mkdir("$datadir/$source")
    retrieval = "rclone --retries=10 copyto r2:rsys/database/collect"
    cmd = "$retrieval/latest $datadir/$source/latest"
    run(`sh -c $cmd`)
    tag = read("$datadir/$source/latest", String)
    if source == "external"
        tables = ["dependencies"]
    else
        tables = ["media", "media_relations"]
    end
    for t in tables
        cmd = "$retrieval/$tag/$(source)_$(t).zstd $datadir/$source/$(source)_$(t).csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $datadir/$source/$(source)_$(t).csv.zstd`)
        rm("$datadir/$source/$(source)_$(t).csv.zstd")
    end
end

function import_media_relations()
    mal_relation_map = Dict(
        "SUMMARY" => "summary",
        "Summary" => "summary",
        "PARENT_STORY" => "parent_story",
        "Parent Story" => "parent_story",
        "PREQUEL" => "prequel",
        "Prequel" => "prequel",
        "SEQUEL" => "sequel",
        "Sequel" => "sequel",
        "ALTERNATIVE_VERSION" => "alternative_version",
        "Alternative Version" => "alternative_version",
        "ALTERNATIVE_SETTING" => "alternative_setting",
        "Alternative Setting" => "alternative_setting",
        "FULL_STORY" => "full_story",
        "Full Story" => "full_story",
        "SIDE_STORY" => "side_story",
        "Side Story" => "side_story",
        "ADAPTATION" => "adaptation",
        "Adaptation" => "adaptation",
        "SPIN_OFF" => "spin_off",
        "Spin-Off" => "spin_off",
        "CHARACTER" => "character",
        "Character" => "character",
        "OTHER" => "other",
        "Other" => "other",
    )

    anilist_relation_map = Dict(
        "SUMMARY" => "summary",
        "PREQUEL" => "prequel",
        "SEQUEL" => "sequel",
        "SPIN_OFF" => "spin_off",
        "CHARACTER" => "character",
        "SIDE_STORY" => "side_story",
        "OTHER" => "other",
        "PARENT" => "parent_story",
        "ADAPTATION" => "adaptation",
        "SOURCE" => "source",
        "ALTERNATIVE" => "alternative_version",
        "COMPILATION" => "compilation",
        "CONTAINS" => "contains",
    )

    kitsu_relation_map = Dict(
        "summary" => "summary",
        "alternative_setting" => "alternative_setting",
        "sequel" => "sequel",
        "parent_story" => "parent_story",
        "prequel" => "prequel",
        "full_story" => "full_story",
        "adaptation" => "adaptation",
        "character" => "character",
        "other" => "other",
        "alternative_version" => "alternative_version",
        "side_story" => "side_story",
        "spinoff" => "spin_off",
    )

    animeplanet_relation_map = Dict("relation" => "unknown")

    relation_maps = Dict(
        "mal" => mal_relation_map,
        "anilist" => anilist_relation_map,
        "kitsu" => kitsu_relation_map,
        "animeplanet" => animeplanet_relation_map,
    )
    for source in keys(relation_maps)
        df =
            CSV.read("$datadir/$source/$(source)_media_relations.csv", DataFrames.DataFrame)
        relation_map = relation_maps[source]
        for x in setdiff(Set(df.relation), keys(relation_map))
            @info "could not parse relation $x from $source"
        end
        df[!, :target_medium] = lowercase.(df.target_medium)
        df[!, :relation] = map(x -> get(relation_map, x, "unknown"), df.relation)
        df[!, :source] .= source
        CSV.write("$datadir/$(source)_media_relations.csv", df)
    end
end

function import_mal(medium)
    source = "mal"
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks = 1)
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        if ismissing(x)
            return missing
        end
        titles = []
        for (k, v) in JSON3.read(x)
            if isempty(v)
                continue
            end
            if k == :synonyms
                append!(titles, v)
            else
                push!(titles, v)
            end
        end
        JSON3.write(titles)
    end
    function mediatype(x)
        typemap = Dict(
            #anime
            "ona" => "ONA",
            "tv" => "TV",
            "ova" => "OVA",
            "special" => "Special",
            "movie" => "Movie",
            "music" => "Music",
            "tv_special" => "TV Special",
            "cm" => "CM",
            "pv" => "PV",
            # manga
            "manhwa" => "Manhwa",
            "manhua" => "Manhua",
            "manga" => "Manga",
            "one_shot" => "One-shot",
            "light_novel" => "Light Novel",
            "doujinshi" => "Doujinshi",
            "novel" => "Novel",
        )
        if ismissing(x) || x == "unknown"
            return missing
        end
        r = get(typemap, x, missing)
        if ismissing(r)
            logerror("import_mal: unknown relation $x")
        end
        r
    end
    function optnum(x)
        if ismissing(x) || x == 0
            return missing
        end
        x
    end
    function status(x, startdate)
        maps = Dict(
            "finished_airing" => "Finished",
            "currently_airing" => "Releasing",
            "not_yet_aired" => ismissing(startdate) ? "TBA" : "Upcoming",
        )
        if ismissing(x)
            return missing
        end
        r = get(maps, x, missing)
        if ismissing(r)
            logerror("import_mal: unknown status $x")
        end
        r
    end
    function season(x)
        if ismissing(x)
            return missing
        end
        json = JSON3.read(x)
        json["season"] * "-" * string(json["year"])
    end
    function mediasource(x)
        if ismissing(x)
            return missing
        end
        source_map = Dict(
            "original" => "Original",
            "manga" => "Manga",
            "Unknown" => missing,
            "game" => "Game",
            "other" => "Other",
            "visual_novel" => "Visual Novel",
            "light_novel" => "Light Novel",
            "novel" => "Novel",
            "web_manga" => "Web Manga",
            "4_koma_manga" => "4-koma Manga",
            "music" => "Music",
            "picture_book" => "Picture Book",
            "mixed_media" => "Mixed Media",
            "book" => "Book",
            "web_novel" => "Web Novel",
            "card_game" => "Card Game",
            "radio" => "Radio",
        )
        if x ∉ keys(source_map)
            logerror("import_mal: unknown source $x")
        end
        get(source_map, x, missing)
    end
    function duration(x)
        if ismissing(x)
            return missing
        end
        x / 60
    end
    ret = DataFrames.DataFrame()
    ret[!, "medium"] = df.medium
    ret[!, "itemid"] = df.itemid
    ret[!, "title"] = df.title
    ret[!, "alternative_titles"] = alternative_titles.(df.alternative_titles)
    ret[!, "mediatype"] = mediatype.(df.media_type)
    ret[!, "startdate"] = df.start_date
    ret[!, "enddate"] = df.end_date
    ret[!, "episodes"] = optnum.(df.num_episodes)
    ret[!, "duration"] = duration.(df.average_episode_duration)
    ret[!, "chapters"] = optnum.(df.num_chapters)
    ret[!, "volumes"] = optnum.(df.num_volumes)
    ret[!, "status"] = [status(x...) for x in zip(df.status, df.start_date)]
    ret[!, "season"] = season.(df.start_season)
    ret[!, "source"] = mediasource.(df.source)
    ret[!, "studios"] = df.studios
    ret[!, "malid"] = df.itemid
    ret[!, "anilistid"] .= missing
    CSV.write("$datadir/$(source)_$(medium).csv", ret)
end

function import_anilist(medium)
    source = "anilist"
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks = 1)
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function title(x)
        json = try
            JSON3.read(x)
        catch
            return x
        end
        for k in ["romaji", "english", "native"]
            if k in keys(json)
                return json[k]
            end
        end
        first(values(json))
    end
    function alternative_titles(title, synonyms)
        r = []
        try
            json = JSON3.read(title)
            append!(r, values(json))
        catch
        end
        try
            json = JSON3.read(synonyms)
            append!(r, json)
        catch
        end
        r = [x for x in r if !isnothing(x)]
        JSON3.write(r)
    end
    function mediatype(x)
        typemap = Dict(
            "SPECIAL" => "Special",
            "ONA" => "ONA",
            "OVA" => "OVA",
            "TV" => "TV",
            "MOVIE" => "Movie",
            "TV_SHORT" => "TV",
            "MUSIC" => "Music",
            "MANGA" => "Manga",
            "ONE_SHOT" => "One-shot",
            "NOVEL" => "Light Novel",
        )
        if ismissing(x)
            return missing
        end
        r = get(typemap, x, missing)
        if ismissing(r)
            logerror("import_anilist: unknown relation $x")
        end
        r
    end
    function date(x)
        if ismissing(x)
            return missing
        end
        json = JSON3.read(x)
        vals = []
        for k in ["year", "month", "day"]
            if isnothing(json[k])
                break
            end
            push!(vals, string(json[k]))
        end
        join(vals, "-")
    end
    function status(x, startdate)
        if ismissing(x)
            return missing
        end
        maps = Dict(
            "FINISHED" => "Finished",
            "NOT_YET_RELEASED" => ismissing(startdate) ? "TBA" : "Upcoming",
            "RELEASING" => "Releasing",
            "CANCELLED" => "Cancelled",
            "HIATUS" => "On Hiatus",
        )
        r = get(maps, x, missing)
        if ismissing(r)
            logerror("import_anilist: unknown status $x")
        end
        r
    end
    function season(year, s)
        if ismissing(year) || ismissing(s)
            return missing
        end
        "$(lowercase(s))-$year"
    end
    function mediasource(x)
        if ismissing(x)
            return missing
        end
        source_map = Dict(
            "ORIGINAL" => "Original",
            "MANGA" => "Manga",
            "VIDEO_GAME" => "Game",
            "VISUAL_NOVEL" => "Visual Novel",
            "LIGHT_NOVEL" => "Light Novel",
            "OTHER" => "Other",
            "NOVEL" => "Novel",
            "WEB_NOVEL" => "Web Novel",
            "MULTIMEDIA_PROJECT" => "Mixed Media",
            "PICTURE_BOOK" => "Picture Book",
            "DOUJINSHI" => "Doujinshi",
            "GAME" => "Game",
            "ANIME" => "Anime",
            "LIVE_ACTION" => "Other",
            "COMIC" => "Other",
        )
        if x ∉ keys(source_map)
            logerror("import_anilist: unknown source $x")
        end
        get(source_map, x, missing)
    end
    ret = DataFrames.DataFrame()
    ret[!, "medium"] = df.medium
    ret[!, "itemid"] = df.itemid
    ret[!, "title"] = title.(df.title)
    ret[!, "alternative_titles"] =
        [alternative_titles(x...) for x in zip(df.title, df.synonyms)]
    ret[!, "mediatype"] = mediatype.(df.mediatype)
    ret[!, "startdate"] = date.(df.startdate)
    ret[!, "enddate"] = date.(df.enddate)
    ret[!, "episodes"] = df.episodes
    ret[!, "duration"] = df.duration
    ret[!, "chapters"] = df.chapters
    ret[!, "volumes"] = df.volumes
    ret[!, "status"] = [status(x...) for x in zip(df.status, df.startdate)]
    ret[!, "season"] = [season(x...) for x in zip(df.seasonyear, df.season)]
    ret[!, "source"] = mediasource.(df.source)
    ret[!, "studios"] = df.studios
    ret[!, "malid"] = df.malid
    ret[!, "anilistid"] = df.itemid
    CSV.write("$datadir/$(source)_$(medium).csv", ret)
end

function import_kitsu(medium)
    source = "kitsu"
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks = 1)
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        try
            json = JSON3.read(x)
            JSON3.write([x for x in values(json) if !isempty(x)])
        catch
            return missing
        end
    end
    function mediatype(x)
        typemap = Dict(
            "movie" => "Movie",
            "TV" => "TV",
            "ONA" => "ONA",
            "OVA" => "OVA",
            "special" => "Special",
            "music" => "Music",
            "doujin" => "Doujinshi",
            "manhwa" => "Manhwa",
            "manga" => "Manga",
            "manhua" => "Manhua",
            "novel" => "Light Novel",
            "oel" => "OEL",
            "oneshot" => "One-shot",
        )
        if ismissing(x)
            return missing
        end
        r = get(typemap, x, missing)
        if ismissing(r)
            logerror("import_kitsu: unknown relation $x")
        end
        r
    end
    function date(x)
        if ismissing(x)
            return missing
        end
        x = string(x)
        while endswith(x, "-01")
            x = chop(x, tail=length("-01"))
        end
        x
    end
    function status(x)
        if ismissing(x)
            return missing
        end
        maps = Dict(
            "finished" => "Finished",
            "current" => "Releasing",
            "tba" => "TBA",
            "unreleased" => "TBA",
            "upcoming" => "Upcoming",
        )
        r = get(maps, x, missing)
        if ismissing(r)
            logerror("import_anilist: unknown status $x")
        end
        r
    end
    ret = DataFrames.DataFrame()
    ret[!, "medium"] = df.medium
    ret[!, "itemid"] = df.itemid
    ret[!, "title"] = df.canonicaltitle
    ret[!, "alternative_titles"] = alternative_titles.(df.titles)
    ret[!, "mediatype"] = mediatype.(df.subtype)
    ret[!, "startdate"] = date.(df.startdate)
    ret[!, "enddate"] = date.(df.enddate)
    ret[!, "episodes"] = df.episodecount
    ret[!, "duration"] = df.episodelength
    ret[!, "chapters"] = df.chaptercount
    ret[!, "volumes"] = df.volumecount
    ret[!, "status"] = status.(df.status)
    ret[!, "season"] .= missing
    ret[!, "source"] .= missing
    ret[!, "studios"] .= missing
    ret[!, "malid"] = df.malid
    ret[!, "anilistid"] = df.anilistid
    CSV.write("$datadir/$(source)_$(medium).csv", ret)
end

function import_animeplanet(medium)
    source = "animeplanet"
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks = 1)
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        if ismissing(x)
            return missing
        else
            return JSON3.write([x])
        end
    end
    function mediatitle(x)
        if ismissing(x)
            return missing
        end
        suffixes = [
            " (Light Novel)",
            " (Novel)",
            " (Pilot)",
            " (Promo)",
            " (One Shot)",
            " (Doujinshi)",
        ]
        for s in suffixes
            if endswith(x, s)
                return chop(x, tail=length(s))
            end
        end
        x
    end
    mediatype_col = Dict("manga" => "title", "anime" => "type")
    function mediatype(x, genres)
        if !ismissing(genres)
            genres = JSON3.read(genres)
            typemap = [
                "light-novels" => "Light Novel",
                "one-shot" => "One-shot",
                "manhwa" => "Manhwa",
                "manhua" => "Manhua",
            ]
            for (k, v) in typemap
                if k in genres
                    return v
                end
            end
        end
        if ismissing(x)
            return missing
        end
        if medium == "manga"
            suffixes = [
                " (Light Novel)" => "Light Novel",
                " (Novel)" => "Novel",
                " (Pilot)" => "One-shot",
                " (Promo)" => "One-shot",
                " (One Shot)" => "One-shot",
                " (Doujinshi)" => "Doujinshi",
            ]
            for (k, v) in suffixes
                if endswith(x, k)
                    return v
                end
            end
            return "Manga"
        elseif medium == "anime"
            key = first(split(strip(x), "\n"))
            map = Dict(
                "TV" => "TV",
                "Web" => "ONA",
                "Movie" => "Movie",
                "TV Special" => "TV Special",
                "OVA" => "OVA",
                "Music Video" => "Music",
                "DVD Special" => "Special",
                "Other" => missing,
                "" => missing,
            )
            return map[key]
        else
            @assert false
        end
    end
    function startdate(x)
        if ismissing(x) || x == "TBA"
            return missing
        end
        fields = split(x, " - ")
        first(fields)
    end
    function enddate(x)
        if ismissing(x) || x == "TBA"
            return missing
        end
        fields = split(x, " - ")
        if length(fields) == 1
            return missing
        end
        @assert length(fields) == 2
        x = fields[end]
        if x == "?"
            return missing
        end
        x
    end
    function episodes(x)
        if ismissing(x) || medium != "anime"
            return missing
        end
        fields = split(strip(x), "\n")
        if length(fields) == 1
            return missing
        end
        key = strip(fields[end])
        regex = Regex("\\(" * """(?s)(.*?)""" * " ep")
        matches = [only(m.captures) for m in eachmatch(regex, key)]
        only(matches)
    end
    function duration(x)
        if ismissing(x) || medium != "anime"
            return missing
        end
        fields = split(strip(x), "\n")
        if length(fields) == 1
            return missing
        end
        key = strip(fields[end])
        regex = Regex(" x " * """(?s)(.*?)""" * " min\\)")
        matches = [only(m.captures) for m in eachmatch(regex, key)]
        if isempty(matches)
            return missing
        end
        only(matches)
    end
    function mangacount(x, name)
        if ismissing(x) || medium != "manga"
            return missing
        end
        regex = Regex(name * "([0-9+]*)")
        matches = [only(m.captures) for m in eachmatch(regex, x)]
        if isempty(matches)
            return missing
        end
        only(matches)
    end
    function mediasource(x)
        if ismissing(x)
            return missing
        end
        genres = JSON3.read(x)
        map = Dict(
            "based-on-a-web-novel" => "Web Novel",
            "based-on-a-manga" => "Manga",
            "based-on-a-light-novel" => "Light Novel",
            "based-on-a-novel" => "Novel",
            "based-on-a-video-game" => "Game",
            "based-on-a-visual-novel" => "Visual Novel",
            "based-on-an-anime" => "Anime",
            "based-on-a-mobile-game" => "Game",
            "based-on-a-doujinshi" => "Doujinshi",
            "based-on-an-eroge" => "Game",
            "based-on-a-fairy-tale" => "Other",
            "based-on-a-4-koma-manga" => "4-koma Manga",
            "based-on-an-otome-game" => "Game",
            "based-on-a-movie" => "Other",
            "based-on-a-webtoon" => "Web Manga",
            "based-on-a-song" => "Other",
            "based-on-a-card-game" => "Card Game",
            "based-on-a-picture-book" => "Picture Book",
            "based-on-a-tv-series" => "Other",
            "based-on-a-cartoon" => "Other",
            "based-on-a-play" => "Other",
            "based-on-a-comic-book" => "Other",
            "based-on-a-religious-text" => missing,
        )
        for k in genres
            if k in keys(map)
                return map[k]
            end
        end
        missing
    end
    ret = DataFrames.DataFrame()
    ret[!, "medium"] = df.medium
    ret[!, "itemid"] = df.itemid
    ret[!, "title"] = mediatitle.(df.title)
    ret[!, "alternative_titles"] = alternative_titles.(df.alttitle)
    ret[!, "mediatype"] = mediatype.(df[:, mediatype_col[medium]], df.genres)
    ret[!, "startdate"] = startdate.(df.year)
    ret[!, "enddate"] = enddate.(df.year)
    ret[!, "episodes"] = episodes.(df.type)
    ret[!, "duration"] = duration.(df.type)
    ret[!, "chapters"] = mangacount.(df.type, "Ch: ")
    ret[!, "volumes"] = mangacount.(df.type, "Vol: ")
    ret[!, "status"] .= missing
    ret[!, "season"] = df.season
    ret[!, "source"] = mediasource.(df.genres)
    ret[!, "studios"] .= df.studios
    ret[!, "malid"] .= missing
    ret[!, "anilistid"] .= missing
    CSV.write("$datadir/$(source)_$(medium).csv", ret)
end

function save_media()
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    Threads.@threads for source in ["external", "mal", "anilist", "kitsu", "animeplanet"]
        download_media(source)
    end
    import_media_relations()
    for m in ["manga", "anime"]
        import_mal(m)
        import_anilist(m)
        import_kitsu(m)
        import_animeplanet(m)        
    end
end

save_media()
