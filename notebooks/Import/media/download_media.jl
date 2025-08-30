import CSV
import CSVFiles
import DataFrames
import Glob
import JSON3
import ProgressMeter: @showprogress
import StatsBase
include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/media"

@kwdef struct Character
    role::Union{String,Missing}
    name::Union{String,Missing}
    gender::Union{String,Missing}
    age::Union{String,Missing}
    description::Union{String,Missing}
end

@kwdef struct Recommendation
    username::String
    itemid::String
    text::String
    count::Int
end

@kwdef struct Review
    username::String
    text::String
    count::Int
    rating::Union{Float64,Missing}
end

@kwdef struct Item
    medium::String
    itemid::String
    malid::Union{String,Missing}
    anilistid::Union{String,Missing}
    title::String
    english_title::Union{String,Missing}
    alternative_titles::Vector{String}
    mediatype::Union{String,Missing}
    status::Union{String,Missing}
    source::Union{String,Missing}
    startdate::Union{String,Missing}
    enddate::Union{String,Missing}
    season::Union{String,Missing}
    episodes::Union{String,Missing}
    volumes::Union{String,Missing}
    chapters::Union{String,Missing}
    duration::Union{Float64,Missing}
    synopsis::Union{String,Missing}
    background::Union{String,Missing}
    characters::Vector{Character}
    genres::Vector{String}
    tags::Vector{String}
    studios::Vector{String}
    authors::Vector{String}
    recommendations::Vector{Recommendation}
    reviews::Vector{Review}
end

function to_dict(s::Union{Character, Recommendation, Review, Item})
    Dict(k => getfield(s, to_dict(k)) for k in fieldnames(typeof(s)))
end
to_dict(s::Vector) = to_dict.(s)
to_dict(s) = s

function read_csv(fn)
    # file contains fields that are too big for the CSV.jl parser
    df = DataFrames.DataFrame(CSVFiles.load(fn, type_detect_rows = 1_000_000))
    if DataFrames.nrow(df) == 0
        return CSV.read(fn, DataFrames.DataFrame) # to get column names
    end
    for col in DataFrames.names(df)
        if eltype(df[!, col]) <: AbstractString
            df[!, col] = replace(df[:, col], "" => missing)
        end
    end
    df
end

write_csv(fn, df) = CSV.write(fn, df)

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
        df = read_csv("$datadir/$source/$(source)_media_relations.csv")
        relation_map = relation_maps[source]
        for x in setdiff(Set(df.relation), keys(relation_map))
            @info "could not parse relation $x from $source"
        end
        df[!, :target_medium] = lowercase.(df.target_medium)
        df[!, :relation] = map(x -> get(relation_map, x, "unknown"), df.relation)
        df[!, :source] .= source
        write_csv("$datadir/$(source)_media_relations.csv", df)
    end
end

function as_dataframe(x::Item)
    DataFrames.DataFrame(
        "medium" => [x.medium],
        "itemid" => [x.itemid],
        "malid" => [x.malid],
        "anilistid" => [x.anilistid],
        "title" => [x.title],
        "english_title" => [x.english_title],
        "alternative_titles" => [JSON3.write(x.alternative_titles)],
        "mediatype" => [x.mediatype],
        "status" => [x.status],
        "source" => [x.source],
        "startdate" => [x.startdate],
        "enddate" => [x.enddate],
        "season" => [x.season],
        "episodes" => [x.episodes],
        "volumes" => [x.volumes],
        "chapters" => [x.chapters],
        "duration" => [x.duration],
        "studios" => [JSON3.write(x.studios)],
    )
end

function import_mal(medium)
    source = "mal"
    df = read_csv("$datadir/$source/$(source)_media.csv")
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        if ismissing(x)
            return []
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
        titles
    end
    function english_title(x)
        if ismissing(x)
            return missing
        end
        get(JSON3.read(x), "en", missing)
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
            logerror("import_mal: unknown mediatype $x")
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
            "finished" => "Finished",
            "currently_publishing" => "Releasing",
            "not_yet_published" => ismissing(startdate) ? "TBA" : "Upcoming",
            "discontinued" => "Cancelled",
            "on_hiatus" => "On Hiatus",
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
    function genre(x)
        if ismissing(x)
            return []
        end
        json = JSON3.read(x)
        vals = [x["name"] for x in json]
        filter!(x -> x ∉ ["Eligible Titles for You Should Read This"], vals)
        vals
    end
    function recommendations(x)
        records = Recommendation[]
        if ismissing(x)
            return records
        end
        r = JSON3.read(x)
        for rec in r
            d = Recommendation(
                username = rec.username,
                itemid = string(rec.itemid),
                text = rec.text,
                count = 1,
            )
            push!(records, d)
        end
        records
    end
    function reviews(x)
        records = Review[]
        if ismissing(x)
            return records
        end
        r = JSON3.read(x)
        for rec in r
            d = Review(
                username = rec.username,
                text = rec.text,
                count = rec.upvotes,
                rating = rec.rating,
            )
            push!(records, d)
        end
        records
    end
    function studios(x)
        if ismissing(x)
            return []
        end
        string.(JSON3.parse(x))
    end
    function authors(x, version)
        if ismissing(x) || version < "5.2.1" # TODO remove version gate
            return []
        end
        ret = []
        for y in JSON3.parse(x)
            name = y["first_name"] * " " * y["last_name"]
            push!(ret, name)
        end
        ret
    end
    optstring(x) = ismissing(x) ? missing : string(x)
    items = Item[]
    for x in eachrow(df)
        item = Item(
            medium = x.medium,
            itemid = string(x.itemid),
            malid = string(x.itemid),
            anilistid = missing,
            title = x.title,
            english_title = english_title(x.alternative_titles),
            alternative_titles = alternative_titles(x.alternative_titles),
            mediatype = mediatype(x.media_type),
            status = status(x.status, x.start_date),
            source = mediasource(x.source),
            startdate = x.start_date,
            enddate = x.end_date,
            season = season(x.start_season),
            episodes = optstring(optnum(x.num_episodes)),
            volumes = optstring(optnum(x.num_volumes)),
            chapters = optstring(optnum(x.num_chapters)),
            duration = duration(x.average_episode_duration),
            synopsis = x.synopsis,
            background = x.background,
            characters = [],
            genres = genre(x.genres),
            tags = [],
            studios = studios(x.studios),
            authors = authors(x.authors, x.version),
            recommendations = recommendations(x.userrec),
            reviews = reviews(x.reviews),
        )
        push!(items, item)
    end
    open("$datadir/$(source)_$(medium).json", "w") do f
        JSON3.write(f, to_dict(items))
    end
    open("$datadir/$(source)_$(medium).csv", "w") do f
        CSV.write(f, reduce(vcat, as_dataframe.(items)))
    end
    items
end

function import_anilist(medium::String)::Vector{Item}
    source = "anilist"
    df = read_csv("$datadir/$source/$(source)_media.csv")
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
        r
    end
    function english_title(title)
        try
            json = JSON3.read(title)
            return something(get(json, "english", missing), missing)
        catch
            return missing
        end
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
            logerror("import_anilist: unknown mediatype $x")
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
    function recommendations(x)
        records = Recommendation[]
        if ismissing(x)
            return records
        end
        records = []
        m = Dict("manga" => "MANGA", "anime" => "ANIME")[medium]
        r = JSON3.read(x)
        for rec in r
            if rec["medium"] != m
                continue
            end
            d = Recommendation(
                username = "",
                itemid = string(rec.itemid),
                text = "",
                count = rec["rating"],
            )
            push!(records, d)
        end
        records
    end
    function reviews(x)
        records = Review[]
        if ismissing(x)
            return records
        end
        r = JSON3.read(x)
        for rec in r
            d = Review(
                username = string(rec.userId),
                rating = rec.score / 10,
                count = rec.rating - (rec.ratingAmount - rec.rating),
                text = rec.summary * "\n" * rec.body,
            )
            push!(records, d)
        end
        records
    end
    function character(x, version)
        records = Character[]
        if ismissing(x) || version < "5.2.0" # TODO remove version check
            return records
        end
        r = JSON3.read(x)
        for rec in r
            if isnothing(rec.node.name.full)
                continue
            end
            d = Character(
                role = rec.role,
                name = rec.node.name.full,
                gender = something(rec.node.gender, missing),
                age = something(rec.node.age, missing),
                description = something(rec.node.description, missing),
            )
            push!(records, d)
        end
        records
    end
    function studios(x)
        if ismissing(x)
            return []
        end
        string.(JSON3.parse(x))
    end
    function authors(x)
        if ismissing(x)
            return []
        end
        ret = []
        for e in JSON3.parse(x)
            if e["role"] in [
                "Story & Art",
                "Story",
                "Art",
                "Director",
                "Original Creator",
                "Original Story",
            ]
                push!(ret, e["name"])
            end
        end
        ret
    end
    function genres(x)
        if ismissing(x)
            return []
        end
        string.(JSON3.parse(x))
    end
    function tags(x)
        if ismissing(x)
            return []
        end
        [y["name"] for y in JSON3.parse(x)]
    end
    optstring(x) = ismissing(x) ? missing : string(x)
    items = Item[]
    for x in eachrow(df)
        item = Item(
            medium = x.medium,
            itemid = string(x.itemid),
            malid = optstring(x.malid),
            anilistid = optstring(x.itemid),
            title = title(x.title),
            english_title = english_title(x.title),
            alternative_titles = alternative_titles(x.title, x.synonyms),
            mediatype = mediatype(x.mediatype),
            status = status(x.status, x.startdate),
            source = mediasource(x.source),
            startdate = date(x.startdate),
            enddate = date(x.enddate),
            season = season(x.seasonyear, x.season),
            episodes = optstring(x.episodes),
            volumes = optstring(x.volumes),
            chapters = optstring(x.chapters),
            duration = x.duration,
            synopsis = x.summary,
            background = missing,
            characters = character(x.characterspeek, x.version),
            genres = genres(x.genres),
            tags = tags(x.tags),
            studios = studios(x.studios),
            authors = authors(x.staffpeek),
            recommendations = recommendations(x.recommendationspeek),
            reviews = reviews(x.reviewspeek),
        )
        push!(items, item)
    end
    open("$datadir/$(source)_$(medium).json", "w") do f
        JSON3.write(f, to_dict(items))
    end
    open("$datadir/$(source)_$(medium).csv", "w") do f
        CSV.write(f, reduce(vcat, as_dataframe.(items)))
    end
    items
end

function import_kitsu(medium)
    source = "kitsu"
    df = read_csv("$datadir/$source/$(source)_media.csv")
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        try
            json = JSON3.read(x)
            return [x for x in values(json) if !isempty(x)]
        catch
            return []
        end
    end
    function english_title(x)
        try
            json = JSON3.read(x)
            return something(get(json, "en"), missing)
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
            logerror("import_kitsu: unknown mediatype $x")
        end
        r
    end
    function date(x)
        if ismissing(x)
            return missing
        end
        x = string(x)
        while endswith(x, "-01")
            x = chop(x, tail = length("-01"))
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
    function genre(x)
        if ismissing(x)
            return []
        end
        JSON3.read(x)
    end
    optstring(x) = ismissing(x) ? missing : string(x)
    items = Item[]
    for x in eachrow(df)
        item = Item(
            medium = x.medium,
            itemid = string(x.itemid),
            malid = optstring(x.malid),
            anilistid = optstring(x.anilistid),
            title = x.canonicaltitle,
            english_title = english_title(x.titles),
            alternative_titles = alternative_titles(x.titles),
            mediatype = mediatype(x.subtype),
            status = status(x.status),
            source = missing,
            startdate = date(x.startdate),
            enddate = date(x.enddate),
            season = missing,
            episodes = optstring(x.episodecount),
            volumes = optstring(x.volumecount),
            chapters = optstring(x.chaptercount),
            duration = x.episodelength,
            synopsis = x.synopsis,
            background = missing,
            characters = [],
            genres = genre(x.genres),
            tags = [],
            studios = [],
            authors = [],
            recommendations = [],
            reviews = [],
        )
        push!(items, item)
    end
    open("$datadir/$(source)_$(medium).json", "w") do f
        JSON3.write(f, to_dict(items))
    end
    open("$datadir/$(source)_$(medium).csv", "w") do f
        CSV.write(f, reduce(vcat, as_dataframe.(items)))
    end
    items
end

function import_animeplanet(medium)
    source = "animeplanet"
    df = read_csv("$datadir/$source/$(source)_media.csv")
    df = filter(x -> x.medium == medium && !ismissing(x.db_last_success_at), df)
    function alternative_titles(x)
        if ismissing(x)
            return []
        else
            return [x]
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
                return chop(x, tail = length(s))
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
    function enddate(date)
        if ismissing(date) || date == "TBA"
            return missing
        end
        fields = split(date, " - ")
        if length(fields) == 1
            return missing
        end
        @assert length(fields) == 2
        date = fields[end]
        if date == "?"
            return missing
        end
        date
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
        m = only(matches)
        if ismissing(m)
            return missing
        end
        parse(Float64, m)
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
    function studios(x)
        if ismissing(x)
            return []
        end
        string.(JSON3.parse(x))
    end
    function tags(x)
        if ismissing(x)
            return []
        end
        vals = string.(JSON3.parse(x))
        titlecase.(replace.(vals, "-" => " "))
    end
    function recommendations(x)
        records = Recommendation[]
        if ismissing(x)
            return records
        end
        r = JSON3.read(x)
        for rec in r
            d = Recommendation(
                username = rec.username,
                itemid = string(rec.itemid),
                text = rec.text,
                count = 1,
            )
            push!(records, d)
        end
        records
    end
    function reviews(x)
        records = Review[]
        if ismissing(x)
            return records
        end
        r = JSON3.read(x)
        for rec in r
            d = Review(
                username = rec.username,
                text = rec.text,
                count = rec.upvotes,
                rating = rec.rating,
            )
            push!(records, d)
        end
        records
    end
    items = Item[]
    for x in eachrow(df)
        item = Item(
            medium = x.medium,
            itemid = string(x.itemid),
            malid = missing,
            anilistid = missing,
            title = mediatitle(x.title),
            english_title = missing,
            alternative_titles = alternative_titles(x.alttitle),
            mediatype = mediatype(x[Symbol(mediatype_col[medium])], x.genres),
            status = missing,
            source = mediasource(x.genres),
            startdate = startdate(x.year),
            enddate = enddate(x.year),
            season = x.season,
            episodes = episodes(x.type),
            volumes = mangacount(x.type, "Vol: "),
            chapters = mangacount(x.type, "Ch: "),
            duration = duration(x.type),
            synopsis = x.summary,
            background = missing,
            characters = [],
            genres = [],
            tags = tags(x.genres),
            studios = studios(x.studios),
            authors = [],
            recommendations = recommendations(x.recommendations),
            reviews = reviews(x.reviews),
        )
        push!(items, item)
    end
    open("$datadir/$(source)_$(medium).json", "w") do f
        JSON3.write(f, to_dict(items))
    end
    open("$datadir/$(source)_$(medium).csv", "w") do f
        CSV.write(f, reduce(vcat, as_dataframe.(items)))
    end
    items
end

function download_items(source::String)
    outdir = "$datadir/users/$source"
    rm(outdir, recursive = true, force = true)
    mkpath(outdir)
    retrieval = "rclone --retries=10 copyto r2:rsys/database/collect"
    cmd = "$retrieval/latest $outdir/latest"
    run(`sh -c $cmd`)
    tag = read("$outdir/latest", String)
    for t in ["user_items"]
        cmd = "$retrieval/$tag/$(source)_$(t).zstd $outdir/$(source)_$(t).csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $outdir/$(source)_$(t).csv.zstd`)
        rm("$outdir/$(source)_$(t).csv.zstd")
    end
    usercol = Dict(
        "mal" => "username",
        "anilist" => "userid",
        "kitsu" => "userid",
        "animeplanet" => "username",
    )[source]
    cmd = "mlr --csv cut -f $usercol,medium,itemid $outdir/$(source)_user_items.csv > $outdir/items.csv"
    run(`sh -c $cmd`)
    rm("$outdir/$(source)_user_items.csv")
    cmd = ("mlr --csv split -n 10000000 --prefix $outdir/items_split $outdir/items.csv")
    run(`sh -c $cmd`)
    rm("$outdir/items.csv")
    fns = Glob.glob("$outdir/items_split*.csv")
    invalid_users = Set()
    for medium in ["manga", "anime"]
        items = Set()
        @showprogress for fn in fns
            df = read_csv(fn)
            mask = df.medium .== medium
            items = union(items, Set(df.itemid[mask]))
        end
        num_items = length(items)
        min_items = 5
        max_items = num_items * 0.8
        @showprogress for fn in fns
            df = read_csv(fn)
            mask = df.medium .== medium
            total_user_counts = StatsBase.countmap(df[:, usercol])
            media_user_counts = StatsBase.countmap(df[:, usercol][mask])
            for k in keys(total_user_counts)
                if total_user_counts[k] < min_items ||
                   get(media_user_counts, k, 0) >= max_items
                    push!(invalid_users, k)
                end
            end
        end
    end
    for medium in ["manga", "anime"]
        @showprogress for (i, fn) in collect(Iterators.enumerate(fns))
            df = read_csv(fn)
            filter!(x -> x[usercol] ∉ invalid_users, df)
            mask = df.medium .== medium
            counts = StatsBase.countmap(df.itemid[mask])
            mdf = DataFrames.DataFrame(
                Dict("itemid" => collect(keys(counts)), "count" => collect(values(counts))),
            )
            mdf[!, "medium"] .= medium
            write_csv("$outdir/$medium.$i.csv", mdf)

        end
        cmd = "cd $outdir && mlr --csv cat $medium.*.csv > $medium.csv && rm $medium.*.csv"
        run(`sh -c $cmd`)
        df = read_csv("$outdir/$medium.csv")
        df = DataFrames.combine(
            DataFrames.groupby(df, [:itemid, :medium]),
            :count => sum => :count,
        )
        write_csv("$outdir/$medium.csv", df)
    end
    cmd = "rm $outdir/items_split*.csv"
    run(`sh -c $cmd`)
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
    for source in reverse(["mal", "anilist", "kitsu", "animeplanet"])
        download_items(source)
    end
end

save_media()