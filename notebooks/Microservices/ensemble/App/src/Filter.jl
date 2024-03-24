# filters generics
display_filter(f) = (df; fn = identity) -> filter(fn ∘ f, df)
inv(f) = (x...) -> f(x...; fn = !)

# filter by related series
# item has been seen
seen = display_filter(x -> x.seen)
# item is related to a seen item
related = display_filter(x -> x.related != 0)
# item is related to a seen item in a different media
cross_related = display_filter(x -> x.is_cross_related != 0)
# item is a recap of a seen item
recap = display_filter(x -> x.is_recap != 0 && x.is_direct_sequel == 0)
# item is a recap of a seen item in a different media
cross_recap = display_filter(x -> x.is_cross_recap != 0 && x.is_direct_sequel == 0)
# item is a sequel that we havent seen the prequel for
dependent = display_filter(x -> x.num_dependencies > 0 && x.is_direct_sequel == 0)

# filter by date
function parse_date(x)
    if x in ["Not available"]
        return nothing
    end
    fields = split(x, " ")
    if length(fields) == 3
        date_format = "u d, Y"
    elseif length(fields) == 2
        date_format = "u Y"
    elseif length(fields) == 1
        date_format = "Y"
    else
        @assert false x
    end
    parsed_date = Dates.DateTime(x, date_format)
    Int(Dates.datetime2unix(parsed_date))
end

function released_after(x, timestamp)
    release_date = parse_date(x.start_date)
    if isnothing(release_date)
        return false
    else
        return timestamp < release_date
    end
end

function released_before(x, timestamp)
    release_date = parse_date(x.start_date)
    if isnothing(release_date)
        return false
    else
        return release_date < timestamp
    end
end

after(year, month = 1, date = 1) = display_filter(
    x -> released_after(x, Dates.datetime2unix(Dates.DateTime(year, month, date))),
)
before(year, month = 1, date = 1) = display_filter(
    x -> released_before(x, Dates.datetime2unix(Dates.DateTime(year, month, date))),
)
status(s) = display_filter(x -> x.status == s)
function released(medium)
    if medium == "anime"
        return inv(status("Not yet aired"))
    elseif medium == "manga"
        return inv(status("Not yet published"))
    else
        @assert false
    end
end

# filter by content
max_episodes(n) = display_filter(x -> x.num_episodes <= n)
search(key::String, col) = display_filter(x -> occursin(lowercase(key), lowercase(x[col])))
search(key, col) = display_filter(x -> x[col] == key)
search(key::Vector, col) = display_filter(x -> x[col] in key)
search(key::String) = search(key, :title)

# filter by score
head(n) = x -> first(x, n)
top(n, field) = x -> first(sort(x, field, rev = true), n)
top(n) = top(n, :score)

# filter columns
function format(df::DataFrame, medium::String, debug::Bool)
    df[!, ""] = 1:length(df.title)
    if debug
        df = DataFrames.select(df, DataFrames.Not([:summary, :tags]))
    else
        if medium == "anime"
            cols = ["", "title", "type", "num_episodes", "start_date", "genres"]
        elseif medium == "manga"
            cols = ["", "title", "type", "status", "start_date", "genres"]
        else
            @assert false
        end
        df = DataFrames.select(df, Symbol.(cols))
    end
    headers = titlecase.(replace.(names(df), "_" => " "))
    headers = replace(
        headers,
        "Title" => titlecase(medium),
        "Num Episodes" => "Episodes",
        "Num Volumes" => "Volumes",
        "Num Chapters" => "Chapters",
    )
    DataFrames.rename!(df, headers)
    df
end

format(medium::String, debug::Bool) = x -> format(x, medium, debug)

function recommend(
    payload::Dict,
    alphas::Dict,
    medium::String,
    source::String;
    M = 1000,
    N = 100,
    debug = false,
    extra_filters = identity,
)
    get_ranking_df(payload, alphas, medium, source) |>
    extra_filters |>
    released(medium) |>
    inv(seen) |>
    inv(recap) |>
    inv(cross_recap) |>
    inv(dependent) |>
    top(1000, :watch) |>
    top(100, :score) |>
    format(medium, debug)
end