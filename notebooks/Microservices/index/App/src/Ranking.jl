to_hyperlink(title, url) = "<a href=\"$url\" target=\"_blank\">$title</a>"

function get_hyperlink(title, links, source)
    if source == "mal"
        search = "myanimelist"
    elseif source == "anilist"
        search = "anilist"
    elseif source == "kitsu"
        search = "kitsu"
    elseif source == "animeplanet"
        search = "anime-planet"
    else
        @assert false
    end

    # try to return the preferred source
    parsed_links = JSON.parse(replace(links, "'" => "\""))
    for s in [search, "myanimelist"]
        for link in parsed_links
            if occursin(s, link)
                return to_hyperlink(title, link)
            end
        end
    end
    title
end

function parse_genres(x)
    genres = JSON.parse(replace(x, "'" => "\"", "_" => " "))
    join(genres, ", ")
end

function get_media(medium::String, source::String)
    df = DataFrame(
        CSV.File(
            get_data_path("processed_data/$medium.csv"),
            ntasks = 1;
            stringtype = String,
        ),
    )
    df.title = get_hyperlink.(df.title, df.links, source)
    df.genres = parse_genres.(df.genres)
    # validate fields
    valid_statuses = Dict(
        "anime" => ["Currently Airing", "Finished Airing", "Not yet aired"],
        "manga" => [
            "Not yet published",
            "Publishing",
            "Discontinued",
            "Finished",
            "On Hiatus",
        ],
    )
    @assert issubset(Set(df.status), Set(valid_statuses[medium]))
    df
end

function prune_media_df(df, medium)
    if medium == "anime"
        series_length = ["num_episodes"]
    elseif medium == "manga"
        series_length = ["num_volumes", "num_chapters"]
    end
    keepcols = vcat(
        ["mediaid", "uid", "title", "type"],
        series_length,
        ["status", "start_date", "end_date", "genres", "tags", "summary"],
    )
    df[:, keepcols]
end

function get_media_df(medium, source)
    media = get_media(medium, source)
    media_to_uid = DataFrame(CSV.File(get_data_path("processed_data/$(medium)_to_uid.csv")))
    df = DataFrames.innerjoin(media_to_uid, media, on = "mediaid" => "$(medium)_id")
    prune_media_df(df, medium)
end

ALL_SOURCES = ["mal", "animeplanet", "kitsu", "animeplanet"]
MEDIA_DFS = Dict(
    "$(medium)_$(source)" => get_media_df(medium, source) 
    for medium in ALL_MEDIUMS for source in ALL_SOURCES
)

# Get rankings

function get_rating_df(payload::Dict, alphas::Dict, medium::String)
    get_alpha(x) = alphas[x]
    rating_df = DataFrame(
        "uid" => 0:num_items(medium)-1,
        "rating" => get_alpha("$medium/Linear/rating"),
        "watch" => get_alpha("$medium/Linear/watch"),
        "plantowatch" => get_alpha("$medium/Linear/plantowatch"),
        "drop" => get_alpha("$medium/Linear/drop"),
        "num_dependencies" => get_alpha("$medium/Nondirectional/Dependencies"),
        "is_sequel" => get_alpha("$medium/Nondirectional/SequelSeries"),
        "is_direct_sequel" => get_alpha("$medium/Nondirectional/DirectSequelSeries"),
        "is_related" => get_alpha("$medium/Nondirectional/RelatedSeries"),
        "is_recap" => get_alpha("$medium/Nondirectional/RecapSeries"),
        "is_cross_recap" => get_alpha("$medium/Nondirectional/CrossRecapSeries"),
        "is_cross_related" => get_alpha("$medium/Nondirectional/CrossRelatedSeries"),
    )
    rating_df[:, "score"] .= (
        rating_df.rating +
        (log.(rating_df.watch) ./ log(10)) +
        0.1 * (log.(rating_df.plantowatch) ./ log(10)) +
        (-max.(rating_df.drop, 0.01) * 10)
    )
    rating_df[:, "seen"] .= false
    seen_df = get_raw_split(payload, medium, [:itemid], nothing)
    rating_df.seen[seen_df.itemid.+1] .= true
    rating_df[:, "ptw"] .= false
    ptw_df = get_split(payload, "plantowatch", medium, [:itemid])
    rating_df.ptw[ptw_df.itemid.+1] .= true
    rating_df.seen[ptw_df.itemid.+1] .= false
    rating_df
end

function get_ranking_df(payload::Dict, alphas::Dict, medium::String, source::String)
    rating_df = get_rating_df(payload, alphas, medium)
    media_df = get_media_df(medium, source)
    DataFrames.innerjoin(media_df, rating_df, on = "uid")
end