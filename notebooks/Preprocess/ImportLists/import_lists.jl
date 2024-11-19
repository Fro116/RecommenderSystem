import CSV
import DataFrames
import Dates
import JSON
import Memoize: @memoize

@kwdef struct RatingsDataset
    source::Vector{Int32}
    medium::Vector{Int32}
    userid::Vector{Int32}
    itemid::Vector{Int32}
    status::Vector{Int32}
    rating::Vector{Float32}
    updated_at::Vector{Float64}
    created_at::Vector{Float64}
    started_at::Vector{Float64}
    finished_at::Vector{Float64}
    update_order::Vector{Int32}
    progress::Vector{Float32}
    progress_volumes::Vector{Float32}
    repeat_count::Vector{Int32}
    priority::Vector{Int32}
    sentiment::Vector{Int32}
end

function subset(x::RatingsDataset, ord)
    RatingsDataset([getfield(x, c)[ord] for c in fieldnames(RatingsDataset)]...)
end

function cat(x::RatingsDataset, y::RatingsDataset)
    RatingsDataset(
        [vcat(getfield(x, c), getfield(y, c)) for c in fieldnames(RatingsDataset)]...,
    )
end

function get_data_path(file)
    joinpath(@__DIR__, "../../../data/$file");
end

function read_csv(x; kw...)
    CSV.read(x, DataFrames.DataFrame; types = String, missingstring = nothing, kw...)
end

function parse_int(x, default = nothing)::Int32
    if isempty(x)
        @assert !isnothing(default)
        return default
    end
    if endswith(x, "+")
        x = x[1:end-1]
    end
    x = parse(Int32, x)
    @assert x >= 0
    x
end

function get_uid_map()
    uid_map = Dict()
    for m in ["manga", "anime"]
        uid_map[m] = Dict()
        df = read_csv(get_data_path("processed_data/$m.csv"); ntasks = 1)
        for s in ["mal", "anilist", "kitsu", "animeplanet"]
            uid_map[m][s] = Dict{String,Int32}()
            for (uid, sourceids) in zip(df[!, :uid] .|> parse_int, df[!, s] .|> JSON.parse)
                for sourceid in sourceids
                    uid_map[m][s][sourceid] = uid
                end
            end
        end
    end
    uid_map
end

function get_progress_map(col_map)
    progress_map = Dict()
    for m in ["manga", "anime"]
        progress_map[m] = Dict()
        col = col_map[m]
        for s in ["mal", "anilist", "kitsu", "animeplanet"]
            progress_map[m][s] = Dict{String,Int32}()
            if isnothing(col)
                continue
            end
            df = read_csv(get_data_path("processed_data/$s.$m.csv"); ntasks = 1)
            for (uid, val) in zip(df[!, :uid], df[!, col])
                progress_map[m][s][uid] = parse_int(val, 0)
            end
        end
    end
    progress_map
end

const SOURCE_MAP =
    Dict{String,Int32}("mal" => 0, "anilist" => 1, "kitsu" => 2, "animeplanet" => 3)
const STATUS_MAP = Dict{String,Int32}(
    "rewatching" => 7,
    "completed" => 6,
    "currently_watching" => 5,
    "on_hold" => 4,
    "planned" => 3,
    "dropped" => 2,
    "wont_watch" => 1,
    "none" => 0,
)
const MEDIUM_MAP = Dict{String,Int32}("manga" => 0, "anime" => 1)
const SENTIMENT_MAP = Dict{String,Int32}("none" => 0, "neutral" => 1)
const UID_MAP = get_uid_map()
const PROGRESS_MAP = get_progress_map(Dict("anime" => "episodes", "manga" => "chapters"))
const PROGRESS_VOLUMES_MAP =
    get_progress_map(Dict("anime" => nothing, "manga" => "volumes"))
const MIN_TS =
    parse(Float64, first(read_csv(get_data_path("processed_data/training.timestamps.csv")).min_ts))
const MAX_TS =
    parse(Float64, first(read_csv(get_data_path("processed_data/training.timestamps.csv")).max_ts))

function parse_timestamp(x::Number, max_valid_ts::Number)::Float64
    if x < MIN_TS || x > max_valid_ts
        return 0
    end
    convert(Float32, (x - MIN_TS) / (MAX_TS - MIN_TS))
end

function parse_timestamp(x::String, max_valid_ts::Number)::Float64
    if isempty(x)
        return 0
    end
    parse_timestamp(parse(Float64, x), max_valid_ts)
end

@memoize function parse_date(x, max_valid_ts)::Float32
    fields = split(x, "-")
    if length(fields) != 3
        return 0
    end
    year = parse_int(fields[1], 1)
    month = parse_int(fields[2], 1)
    day = parse_int(fields[3], 1)
    try
        dt = Dates.DateTime(year, month, day)
        return parse_timestamp(Dates.datetime2unix(dt), max_valid_ts)
    catch e
        return 0
    end
end

@memoize function process_rating(score)::Float32
    score = parse(Float32, score)
    if !(0 <= score <= 10)
        @warn "invalid score $score"
        return 0.0f0
    end
    score
end

function parse_progress(medium, source, df, progress_col, uid_col)::Vector{Float32}
    clamp.(
        (df[:, progress_col] .|> x -> parse_int(x, 0)) ./
        (df[:, uid_col] .|> x -> get(PROGRESS_MAP[medium][source], x, 0)),
        0,
        1,
    ) .|> x -> isnan(x) ? 0.0f0 : x .|> x -> convert(Float32, x)
end

function parse_sentiment(x::String)::Int32
    isempty(x) ? 0 : 1
end

function import_mal(medium, userid_map, max_valid_ts, df)
    @memoize function process_status(status)
        mal_status_map = Dict(
            "completed" => "completed",
            "watching" => "currently_watching",
            "plan_to_watch" => "planned",
            "reading" => "currently_watching",
            "plan_to_read" => "planned",
            "on_hold" => "on_hold",
            "dropped" => "dropped",
            "" => "none",
        )
        STATUS_MAP[mal_status_map[status]]
    end

    @memoize function parse_repeat(x)
        repeat_map = Dict("True" => "rewatching", "False" => "none")
        STATUS_MAP[repeat_map[x]]
    end

    parse_timestamp(x) = Main.parse_timestamp(x, max_valid_ts)
    parse_date(x) = Main.parse_date(x, max_valid_ts)

    s = "mal"
    RatingsDataset(
        source = fill(SOURCE_MAP[s], DataFrames.nrow(df)),
        medium = fill(MEDIUM_MAP[medium], DataFrames.nrow(df)),
        userid = df[!, :username] .|> x -> get(userid_map, x, 0),
        itemid = df[!, :uid] .|> x -> get(UID_MAP[medium][s], x, 0),
        status = max.(df[!, :status] .|> process_status, df[!, :repeat] .|> parse_repeat),
        rating = df[!, :score] .|> process_rating,
        updated_at = df[!, :updated_at] .|> x -> parse_int(x, 0) .|> parse_timestamp,
        created_at = zeros(Float32, DataFrames.nrow(df)),
        started_at = df[!, :started_at] .|> parse_date,
        finished_at = df[!, :completed_at] .|> parse_date,
        update_order = zeros(Int32, DataFrames.nrow(df)),
        progress = parse_progress(medium, s, df, :progress, :uid),
        progress_volumes = parse_progress(medium, s, df, :progress_volumes, :uid),
        repeat_count = zeros(Int32, DataFrames.nrow(df)),
        priority = zeros(Int32, DataFrames.nrow(df)),
        sentiment = zeros(Int32, DataFrames.nrow(df)),
    )
end

function import_anilist(medium, userid_map, max_valid_ts, df)
    @memoize function process_status(status)
        anilist_status_map = Dict(
            "REPEATING" => "rewatching",
            "COMPLETED" => "completed",
            "CURRENT" => "currently_watching",
            "PLANNING" => "planned",
            "PAUSED" => "on_hold",
            "DROPPED" => "dropped",
        )
        STATUS_MAP[anilist_status_map[status]]
    end

    parse_timestamp(x) = Main.parse_timestamp(x, max_valid_ts)
    parse_date(x) = Main.parse_date(x, max_valid_ts)

    s = "anilist"
    RatingsDataset(
        source = fill(SOURCE_MAP[s], DataFrames.DataFrames.nrow(df)),
        medium = fill(MEDIUM_MAP[medium], DataFrames.DataFrames.nrow(df)),
        userid = df[!, :username] .|> x -> get(userid_map, x, 0),
        itemid = df[!, :anilistid] .|> x -> get(UID_MAP[medium][s], x, 0),
        status = df[!, :status] .|> process_status,
        rating = df[!, :score] .|> process_rating,
        updated_at = df[!, :updated_at] .|> parse_timestamp,
        created_at = df[!, :created_at] .|> parse_timestamp,
        started_at = df[!, :started_at] .|> parse_date,
        finished_at = df[!, :completed_at] .|> parse_date,
        update_order = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        progress = parse_progress(medium, s, df, :progress, :anilistid),
        progress_volumes = parse_progress(medium, s, df, :progress_volumes, :anilistid),
        repeat_count = df[!, :repeat] .|> parse_int,
        priority = df[!, :priority] .|> parse_int,
        sentiment = df[!, :notes] .|> parse_sentiment,
    )
end

function import_kitsu(medium, userid_map, max_valid_ts, df)
    @memoize function process_status(status)
        kitsu_status_map = Dict(
            "completed" => "completed",
            "current" => "currently_watching",
            "dropped" => "dropped",
            "on_hold" => "on_hold",
            "planned" => "planned",
        )
        STATUS_MAP[kitsu_status_map[status]]
    end

    @memoize function parse_repeat(x)
        repeat_map = Dict("True" => "rewatching", "False" => "none")
        STATUS_MAP[repeat_map[x]]
    end

    parse_timestamp(x) = Main.parse_timestamp(x, max_valid_ts)
    parse_date(x) = Main.parse_date(x, max_valid_ts)

    s = "kitsu"
    RatingsDataset(
        source = fill(SOURCE_MAP[s], DataFrames.DataFrames.nrow(df)),
        medium = fill(MEDIUM_MAP[medium], DataFrames.DataFrames.nrow(df)),
        userid = df[!, :username] .|> x -> get(userid_map, x, 0),
        itemid = df[!, :kitsuid] .|> x -> get(UID_MAP[medium][s], x, 0),
        status = max.(df[!, :status] .|> process_status, df[!, :repeat] .|> parse_repeat),
        rating = df[!, :score] .|> process_rating,
        updated_at = df[!, :updated_at] .|> parse_timestamp,
        created_at = df[!, :created_at] .|> parse_timestamp,
        started_at = df[!, :started_at] .|> parse_timestamp,
        finished_at = df[!, :finished_at] .|> parse_timestamp,
        update_order = zeros(Int, DataFrames.DataFrames.nrow(df)),
        progress = parse_progress(medium, s, df, :progress, :kitsuid),
        progress_volumes = parse_progress(medium, s, df, :volumes_owned, :kitsuid),
        repeat_count = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        priority = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        sentiment = df[!, :notes] .|> parse_sentiment,
    )
end

function import_animeplanet(medium, userid_map, max_valid_ts, df)
    @memoize function process_status(status)
        animeplanet_status_map = Dict(
            "1" => "completed",
            "2" => "currently_watching",
            "3" => "dropped",
            "4" => "planned",
            "5" => "on_hold",
            "6" => "wont_watch",
        )
        STATUS_MAP[animeplanet_status_map[status]]
    end

    parse_timestamp(x) = Main.parse_timestamp(x, max_valid_ts)
    parse_date(x) = Main.parse_date(x, max_valid_ts)

    s = "animeplanet"
    RatingsDataset(
        source = fill(SOURCE_MAP[s], DataFrames.DataFrames.nrow(df)),
        medium = fill(MEDIUM_MAP[medium], DataFrames.DataFrames.nrow(df)),
        userid = df[!, :username] .|> x -> get(userid_map, x, 0),
        itemid = df[!, :url] .|> x -> get(UID_MAP[medium][s], x, 0),
        status = df[!, :status] .|> process_status,
        rating = df[!, :score] .|> process_rating,
        updated_at = df[!, :updated_at] .|> parse_timestamp,
        created_at = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        started_at = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        finished_at = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        update_order = df[!, :item_order] .|> parse_int,
        progress = parse_progress(medium, s, df, :progress, :url),
        progress_volumes = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        repeat_count = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        priority = zeros(Int32, DataFrames.DataFrames.nrow(df)),
        sentiment = zeros(Int32, DataFrames.DataFrames.nrow(df)),
    )
end

function import_list(medium, source, userid_map, max_valid_ts, df)::RatingsDataset
    source_map = Dict(
        "mal" => import_mal,
        "anilist" => import_anilist,
        "kitsu" => import_kitsu,
        "animeplanet" => import_animeplanet,
    )
    df = source_map[source](medium, userid_map, max_valid_ts, df)

    # drop invalid users and items
    df = subset(df, (df.itemid .!= 0) .&& (df.userid .!= 0))
    # drop duplicate rows
    if length(df.userid) > 0
        df = subset(df, sortperm(collect(zip(df.userid, df.itemid))))
        dups = collect(zip(df.userid, df.itemid))
        df = subset(df, BitVector([dups[1:end-1] .!= dups[2:end]; true]))
    end
    # sort by update time
    df = subset(df, sortperm(collect(zip(df.userid, df.updated_at, df.update_order))))
    df
end