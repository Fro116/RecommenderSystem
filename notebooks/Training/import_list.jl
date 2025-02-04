import CSV
import Dates
import DataFrames
import Memoize: @memoize

const STATUS_MAP = Dict{String,Int32}(
    "none" => 0,
    "wont_watch" => 1,
    "dropped" => 2,
    "planned" => 3,
    "on_hold" => 4,
    "currently_watching" => 5,
    "completed" => 6,
    "rewatching" => 7,
)

const MEDIUM_MAP = Dict{String,Int32}("manga" => 0, "anime" => 1)

const SOURCE_MAP = Dict{String,Int32}(
    "mal" => 0,
    "anilist" => 1,
    "kitsu" => 2,
    "animeplanet" => 3,
)

const mediadir = "../../data/training"

@memoize function get_media(medium)
    df = CSV.read(
        "$mediadir/$medium.csv",
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
    )
    d = Dict()
    for i = 1:DataFrames.nrow(df)
        d[(df.source[i], df.itemid[i])] = Dict(
            "episodes" => df.episodes[i],
            "chapters" => df.chapters[i],
            "volumes" => df.volumes[i],
            "distinctid" => df.distinctid[i],
            "matchedid" => df.matchedid[i],
        )
    end
    d
end

function read_csv(fn)
    # CSV.read can't handle rows with >1m characters
    lines = readlines(fn)
    rows = Vector{Any}(undef, length(lines))
    Threads.@threads for i = 1:length(lines)
        rows[i] = split(lines[i], r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
    end
    colnames = rows[1]
    records = rows[2:end]
    df = DataFrames.DataFrame(
        Dict(colnames[i] => [x[i] for x in records] for i = 1:length(colnames)),
    )
    df[:, colnames]
end

function decompress(x)
    MsgPack.unpack(
        CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(hex2bytes(x[3:end]))),
    )
end

function get_progress(source, medium, itemid, episodes, chapters, volumes)::Float32
    df = get_media(medium)
    k = (source, string(itemid))
    if k ∉ keys(df)
        return 0
    end
    isvalid(x) = !ismissing(x) && !isnothing(x) && x != 0
    progress = 0
    if isvalid(episodes) && isvalid(df[k]["episodes"])
        progress = max(progress, episodes / df[k]["episodes"])
    elseif isvalid(chapters) && isvalid(df[k]["chapters"])
        progress = max(progress, chapters / df[k]["chapters"])
    elseif isvalid(volumes) && isvalid(df[k]["volumes"])
        progress = max(progress, volumes / df[k]["volumes"])
    end
    min(progress, 1.0)
end

function create_item!(source, medium, itemid, item)
    if item["medium"] ∉ values(MEDIUM_MAP)
        logerror("invalid medium for $item")
        item["medium"] = 0
    end
    if item["status"] ∉ values(STATUS_MAP)
        logerror("invalid status for $item")
        item["status"] = 0
    end
    if item["rating"] > 10 || item["rating"] < 0
        logerror("invalid rating for $item")
        item["rating"] = 0
    end
    if item["update_order"] < 0
        logerror("invalid update_order for $item")
        item["update_order"] = 0
    end
    if item["updated_at"] <= 0
        item["updated_at"] = 0
    elseif item["updated_at"] < Dates.datetime2unix(Dates.DateTime(2000, 1, 1)) ||
           item["updated_at"] > time() + 86400
        logerror("invalid updated_at for $item")
        item["updated_at"] = 0
    end
    if item["progress"] < 0 || item["progress"] > 1
        logerror("invalid progress for $item")
        item["progress"] = 0
    end
    d = get(get_media(medium), (source, string(itemid)), Dict())
    item["distinctid"] = get(d, "distinctid", 0)
    item["matchedid"] = get(d, "matchedid", 0)
end

function import_mal_user(data)
    source = "mal"
    user = Dict("source" => SOURCE_MAP[source])
    items = []
    status_map = Dict(
        "completed" => "completed",
        "watching" => "currently_watching",
        "plan_to_watch" => "planned",
        "reading" => "currently_watching",
        "plan_to_read" => "planned",
        "on_hold" => "on_hold",
        "dropped" => "dropped",
        nothing => "none",
    )
    for x in data["items"]
        item = Dict(
            "medium" => MEDIUM_MAP[x["medium"]],
            "status" => STATUS_MAP[status_map[x["status"]]],
            "rating" => convert(Int32, x["score"]),
            "updated_at" =>
                isnothing(x["updated_at"]) ? 0 :
                Dates.datetime2unix(
                    Dates.DateTime(x["updated_at"], "yyyy-mm-ddTHH:MM:SS+00:00"),
                ),
            "update_order" => 0,
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                x["num_volumes_read"],
            ),
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by=x -> (x["updated_at"], x["update_order"]))
    Dict("user" => user, "items" => items)
end

function import_anilist_user(data)
    source = "anilist"
    user = Dict("source" => SOURCE_MAP[source])
    items = []
    status_map = Dict(
        "REPEATING" => "rewatching",
        "COMPLETED" => "completed",
        "CURRENT" => "currently_watching",
        "PLANNING" => "planned",
        "PAUSED" => "on_hold",
        "DROPPED" => "dropped",
        nothing => "none",
    )
    for x in data["items"]
        item = Dict(
            "medium" => MEDIUM_MAP[x["medium"]],
            "status" => STATUS_MAP[status_map[x["status"]]],
            "rating" => convert(Float32, x["score"]),
            "updated_at" => convert(Float32, something(x["updatedat"], 0)),
            "update_order" => 0,
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                x["progressvolumes"],
            ),
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by=x -> (x["updated_at"], x["update_order"]))
    Dict("user" => user, "items" => items)
end

function import_kitsu_user(data)
    source = "kitsu"
    user = Dict("source" => SOURCE_MAP[source])
    items = []
    status_map = Dict(
        "completed" => "completed",
        "current" => "currently_watching",
        "dropped" => "dropped",
        "on_hold" => "on_hold",
        "planned" => "planned",
        nothing => "none",
    )
    for x in data["items"]
        item = Dict(
            "medium" => MEDIUM_MAP[x["medium"]],
            "status" => STATUS_MAP[status_map[x["status"]]],
            "rating" => convert(Float32, something(x["ratingtwenty"], 0)) / 2,
            "updated_at" =>
                isnothing(x["updatedat"]) ? 0 :
                Dates.datetime2unix(
                    Dates.DateTime(x["updatedat"], "yyyy-mm-ddTHH:MM:SS.sssZ"),
                ),
            "update_order" => 0,
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                nothing,
            ),
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by=x -> (x["updated_at"], x["update_order"]))
    Dict("user" => user, "items" => items)
end

function import_animeplanet_user(data)
    source = "animeplanet"
    user = Dict("source" => SOURCE_MAP[source])
    items = []
    status_map = Dict(
        1 => "completed",
        2 => "currently_watching",
        3 => "dropped",
        4 => "planned",
        5 => "on_hold",
        6 => "wont_watch",
        nothing => "none",
    )
    for x in data["items"]
        item = Dict(
            "medium" => MEDIUM_MAP[x["medium"]],
            "status" => STATUS_MAP[status_map[x["status"]]],
            "rating" => convert(Float32, something(x["score"], 0)) * 2,
            "updated_at" => something(x["updated_at"], 0),
            "update_order" => convert(Int32, something(x["item_order"], 0)),
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                nothing,
            ),
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by=x -> (x["updated_at"], x["update_order"]))
    Dict("user" => user, "items" => items)
end

function import_user(source, data)
    if source == "mal"
        return import_mal_user(data)
    elseif source == "anilist"
        return import_anilist_user(data)
    elseif source == "kitsu"
        return import_kitsu_user(data)
    elseif source == "animeplanet"
        return import_animeplanet_user(data)
    else
        @assert false
    end
end
