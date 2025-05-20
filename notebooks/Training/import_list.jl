import CodecZstd
import CSV
import Dates
import DataFrames
import JSON3
import Memoize: @memoize
import MsgPack

const STATUS_MAP = Dict{String,Int32}(
    "none" => 0,
    "wont_watch" => 1,
    "dropped" => 2,
    "deleted" => 3,
    "on_hold" => 4,
    "planned" => 5,
    "currently_watching" => 6,
    "completed" => 7,
    "rewatching" => 8,
)

const MEDIUM_MAP = Dict{String,Int32}("manga" => 0, "anime" => 1)

const SOURCE_MAP =
    Dict{String,Int32}("mal" => 0, "anilist" => 1, "kitsu" => 2, "animeplanet" => 3)
const GENDER_MAP = Dict{String,Int32}("male" => 0, "female" => 1, "other" => 2)

@memoize function get_media_progress(medium)
    df = CSV.read(
        "$datadir/$medium.csv",
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
        ntasks = 1,
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

function decompress(x::AbstractString)
    MsgPack.unpack(
        CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(hex2bytes(x[3:end]))),
    )
end

function decompress(x::Vector)
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(x)))
end

function get_progress(source, medium, itemid, episodes, chapters, volumes)::Float32
    df = get_media_progress(medium)
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
    for k in ["updated_at", "history_min_ts", "history_max_ts"]
        if item[k] <= 0
            item[k] = 0
        elseif item[k] < Dates.datetime2unix(Dates.DateTime(2000, 1, 1)) ||
               item[k] > time() + 86400
            logerror("invalid $k for $item $(item[k])")
            item[k] = 0
        end
    end
    if item["progress"] < 0 || item["progress"] > 1
        logerror("invalid progress for $item")
        item["progress"] = 0
    end
    d = get(get_media_progress(medium), (source, string(itemid)), Dict())
    item["distinctid"] = get(d, "distinctid", 0)
    item["matchedid"] = get(d, "matchedid", 0)
    @assert (item["history_tag"] in ["infer", "delete"]) ||
            Dates.datetime2unix(Dates.DateTime(item["history_tag"], "yyyymmdd")) <
            time() + 86400
end

function import_mal_profile(data, reftime)
    function mal_last_online(datestr, reftime)
        if isnothing(datestr) || datestr in ["Never"]
            return nothing
        end
        if endswith(datestr, "hours ago") || endswith(datestr, "hour ago")
            h = parse(Int, first(split(datestr)))
            return reftime - h * 3600
        end
        if endswith(datestr, "minutes ago") || endswith(datestr, "minute ago")
            m = parse(Int, first(split(datestr)))
            return reftime - m * 60
        end
        if startswith(datestr, "Yesterday") || startswith(datestr, "Today")
            dt = Dates.unix2datetime(reftime)
            if startswith(datestr, "Yesterday")
                dt -= Dates.Day(1)
                str = datestr[length("Yesterday, ")+1:end]
            elseif startswith(datestr, "Today")
                str = datestr[length("Today, ")+1:end]
            else
                @assert false
            end
            y = Dates.Year(dt).value
            m = Dates.Month(dt).value
            d = Dates.Day(dt).value
            str = "$y $m $d, $str"
            f = Dates.DateFormat("yyyy m d, I:M p")
            try
                return Dates.datetime2unix(Dates.DateTime(str, f))
            catch
                nothing
            end
        end
        try
            f = Dates.DateFormat("u d, y I:M p")
            return Dates.datetime2unix(Dates.DateTime(datestr, f))
        catch
            nothing
        end
        try
            f = Dates.DateFormat("yyyy u d, I:M p")
            y = Dates.Year(Dates.unix2datetime(reftime)).value
            str = "$y $datestr"
            return Dates.datetime2unix(Dates.DateTime(str, f))
        catch
            nothing
        end
        logerror("mal_last_online: failed to parse $datestr $reftime")
        nothing
    end

    function mal_gender(x)
        if isnothing(x)
            return nothing
        end
        gender = Dict("Male" => "male", "Female" => "female", "Non-Binary" => "other")[x]
        GENDER_MAP[gender]
    end

    function mal_date(x)
        if isnothing(x)
            return nothing
        end
        for f in Dates.DateFormat.(["u d, yyyy", "u yyyy", "d, yyyy", "yyyy"])
            try
                return Dates.datetime2unix(Dates.DateTime(x, f))
            catch
                nothing
            end
        end
        nothing
    end
    user = data["user"]
    Dict(
        "source" => SOURCE_MAP["mal"],
        "username" => user["username"],
        "last_online" => mal_last_online(user["last_online"], reftime),
        "avatar" => user["avatar"],
        "banner_image" => nothing,
        "gender" => mal_gender(user["gender"]),
        "birthday" => mal_date(user["birthday"]),
        "accessed_at" => reftime,
        "created_at" => mal_date(user["joined"]),
        "location" => user["location"],
        "about" => user["about"],
    )
end

function import_mal_user(data, reftime)
    source = "mal"
    user = import_mal_profile(data, reftime)
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
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                x["num_volumes_read"],
            ),
            "history_min_ts" => x["history_min_ts"],
            "history_max_ts" => x["history_max_ts"],
            "history_tag" => x["history_tag"],
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    Dict("user" => user, "items" => items)
end

function import_anilist_profile(data, reftime)
    function anilist_last_online(data)
        vals = []
        push!(vals, data["user"]["updatedat"])
        for x in data["items"]
            push!(vals, x["updatedat"])
        end
        vals = filter(x -> !isnothing(x) && x > 0, vals)
        if isempty(vals)
            return nothing
        end
        maximum(vals)
    end
    function anilist_image(x)
        if isnothing(x)
            return nothing
        end
        json = JSON3.read(x)
        if isempty(json)
            return nothing
        end
        for k in ["large", "medium"]
            if k in keys(json)
                return json[k]
            end
        end
        nothing
    end
    function anilist_avatar(x)
        default_imgs =
            ["https://s4.anilist.co/file/anilistcdn/user/avatar/large/default.png"]
        if x in default_imgs
            return nothing
        end
        x
    end
    user = data["user"]
    Dict(
        "source" => SOURCE_MAP["anilist"],
        "username" => user["username"],
        "last_online" => anilist_last_online(data),
        "avatar" => anilist_avatar(anilist_image(user["avatar"])),
        "banner_image" => user["bannerimage"],
        "gender" => nothing,
        "birthday" => nothing,
        "accessed_at" => reftime,
        "created_at" => user["createdat"],
        "location" => nothing,
        "about" => user["about"],
    )
end

function import_anilist_user(data, reftime)
    source = "anilist"
    user = import_anilist_profile(data, reftime)
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
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                x["progressvolumes"],
            ),
            "history_min_ts" => x["history_min_ts"],
            "history_max_ts" => x["history_max_ts"],
            "history_tag" => x["history_tag"],
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    Dict("user" => user, "items" => items)
end

function import_kitsu_profile(data, reftime)
    function import_kitsu_time(x)
        if isnothing(x)
            return nothing
        end
        @assert x[end] == 'Z'
        Dates.datetime2unix(Dates.DateTime(x[1:end-1], "yyyy-mm-ddTHH:MM:SS.sss"))
    end
    function kitsu_last_online(data)
        vals = []
        push!(vals, import_kitsu_time(data["user"]["updatedat"]))
        for x in data["items"]
            push!(vals, import_kitsu_time(x["updatedat"]))
        end
        vals = filter(x -> !isnothing(x) && x > 0, vals)
        if isempty(vals)
            return nothing
        end
        maximum(vals)
    end
    function kitsu_image(x)
        if isnothing(x)
            return nothing
        end
        json = JSON3.read(x)
        if isempty(json)
            return nothing
        end
        for k in ["original", "large", "medium"]
            if k in keys(json)
                return json[k]
            end
        end
        nothing
    end
    function kitsu_gender(x)
        if isnothing(x)
            return nothing
        end
        if x in ["male", "female"]
            gender = x
        else
            gender = "other"
        end
        GENDER_MAP[gender]
    end
    function kitsu_birthday(x)
        if isnothing(x)
            return nothing
        end
        for f in Dates.DateFormat.(["yyyy-mm-dd"])
            try
                return Dates.datetime2unix(Dates.DateTime(x, f))
            catch
                nothing
            end
        end
        nothing
    end
    user = data["user"]
    Dict(
        "source" => SOURCE_MAP["kitsu"],
        "username" => user["name"],
        "last_online" => kitsu_last_online(data),
        "avatar" => kitsu_image(user["avatar"]),
        "banner_image" => kitsu_image(user["coverimage"]),
        "gender" => kitsu_gender(user["gender"]),
        "birthday" => kitsu_birthday(user["birthday"]),
        "accessed_at" => reftime,
        "created_at" => import_kitsu_time(user["createdat"]),
        "location" => user["location"],
        "about" => user["about"],
    )
end

function import_kitsu_user(data, reftime)
    function import_kitsu_time(x)
        if isnothing(x)
            return 0
        end
        @assert x[end] == 'Z'
        Dates.datetime2unix(Dates.DateTime(x[1:end-1], "yyyy-mm-ddTHH:MM:SS.sss"))
    end
    source = "kitsu"
    user = import_kitsu_profile(data, reftime)
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
            "updated_at" => import_kitsu_time(x["updatedat"]),
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                nothing,
            ),
            "history_min_ts" => x["history_min_ts"],
            "history_max_ts" => x["history_max_ts"],
            "history_tag" => x["history_tag"],
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    Dict("user" => user, "items" => items)
end

function import_animeplanet_profile(data, reftime)
    function animeplanet_last_online(datestr, reftime)
        if isnothing(datestr)
            return nothing
        end
        if startswith(datestr, "Currently online")
            return reftime
        end
        if datestr in ["Hidden", "Last online: never"]
            return nothing
        end
        if !startswith(datestr, "Last online ")
            logerror("animeplanet_last_online: $datestr")
            return nothing
        end
        datestr = datestr[length("Last online ")+1:end]
        if endswith(datestr, "hours ago") || endswith(datestr, "hour ago")
            h = parse(Int, first(split(datestr)))
            return reftime - h * 3600
        end
        if endswith(datestr, "mins ago") || endswith(datestr, "min ago")
            m = parse(Int, first(split(datestr)))
            return reftime - m * 60
        end
        try
            f = Dates.DateFormat("u d, y")
            return Dates.datetime2unix(Dates.DateTime(datestr, f))
        catch
            nothing
        end
        logerror("animeplanet_last_online: failed to parse $datestr $reftime")
        nothing
    end
    function animeplanet_gender(x)
        if isnothing(x)
            return nothing
        end
        age, genderstr = split(x, " / ")
        if genderstr == "?"
            return nothing
        end
        gender = Dict("M" => "male", "F" => "female", "O" => "other")[genderstr]
        GENDER_MAP[gender]
    end

    function animeplanet_birthday(x, reftime)
        if isnothing(x)
            return nothing
        end
        age, genderstr = split(x, " / ")
        if age == "?"
            return nothing
        end
        reftime - 86400 * parse(Int, age)
    end
    function animeplanet_created_at(datestr)
        if isnothing(datestr)
            return nothing
        end
        if !startswith(datestr, "Joined ")
            logerror(datestr)
            @assert false
            return nothing
        end
        datestr = datestr[length("Joined ")+1:end]
        try
            f = Dates.DateFormat("u d, y")
            return Dates.datetime2unix(Dates.DateTime(datestr, f))
        catch
            nothing
        end
        logerror("animeplanet_created_at: failed to parse $datestr $reftime")
        nothing
    end
    user = data["user"]
    Dict(
        "source" => SOURCE_MAP["animeplanet"],
        "username" => user["username"],
        "last_online" => animeplanet_last_online(user["last_online"], reftime),
        "avatar" => user["avatar"],
        "banner_image" => user["banner_image"],
        "gender" => animeplanet_gender(user["age"]),
        "birthday" => animeplanet_birthday(user["age"], reftime),
        "accessed_at" => reftime,
        "created_at" => animeplanet_created_at(user["joined"]),
        "location" => user["location"],
        "about" => user["about"],
    )
end

function import_animeplanet_user(data, reftime)
    source = "animeplanet"
    user = import_animeplanet_profile(data, reftime)
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
            "progress" => get_progress(
                source,
                x["medium"],
                x["itemid"],
                x["progress"],
                x["progress"],
                nothing,
            ),
            "history_min_ts" => x["history_min_ts"],
            "history_max_ts" => x["history_max_ts"],
            "history_tag" => x["history_tag"],
        )
        create_item!(source, x["medium"], x["itemid"], item)
        push!(items, item)
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    Dict("user" => user, "items" => items)
end

function import_profile(source, data, reftime)
    fns = Dict(
        "mal" => import_mal_profile,
        "anilist" => import_anilist_profile,
        "kitsu" => import_kitsu_profile,
        "animeplanet" => import_animeplanet_profile,
    )
    fns[source](data, reftime)
end

function annotate_with_last_state!(user)
    # remove acausal deleted items
    items = user["items"]
    while !isempty(items) && items[end]["history_tag"] == "delete"
        items = items[1:end-1]
    end
    # set metrics for custom tags
    deleted_status = 3
    for x in items
        tag = x["history_tag"]
        if tag == "infer"
            x["status"] = 0
            x["rating"] = 0
            x["progress"] = 0
        elseif tag == "delete"
            x["status"] = deleted_status
            x["rating"] = 0
        end
    end
    # record last item state
    snapshot = Dict()
    for x in items
        k = (x["medium"], x["matchedid"])
        if k in keys(snapshot)
            x["history_status"], x["history_rating"] = snapshot[k]
        else
            x["history_status"] = nothing
            x["history_rating"] = nothing
        end
        snapshot[k] = (x["status"], x["rating"])
    end
    user["items"] = items
end

function import_user(source, data, reftime)
    fns = Dict(
        "mal" => import_mal_user,
        "anilist" => import_anilist_user,
        "kitsu" => import_kitsu_user,
        "animeplanet" => import_animeplanet_user,
    )
    user = fns[source](data, reftime)
    annotate_with_last_state!(user)
    user
end
