import CSV
import DataFrames
import Dates
import JSON3
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random
import StatsBase
import StringDistances
include("../../julia_utils/multithreading.jl")
include("../../julia_utils/stdout.jl")
include("common.jl")

struct TitleType
    title::String
    alttitles::Vector{String}
end

struct DateType
    date::Dates.Date
    resolution::Int
end

struct LengthType
    length::Real
    plus::Bool
end

function to_titletype(title, alternative_titles)
    if ismissing(title)
        return missing
    end
    title = lowercase(title)
    alttitles = []
    if !ismissing(alternative_titles)
        for x in JSON3.read(alternative_titles)
            if !ismissing(x)
                push!(alttitles, lowercase(x))
            end
        end
    end
    TitleType(title, alttitles)
end

function to_datetype(x)
    if ismissing(x)
        return missing
    end
    fields = split(string(x), "-")
    while !isempty(fields)
        try
            Dates.Date(join(fields, "-"))
            break
        catch
            pop!(fields)
        end
    end
    if isempty(fields)
        return missing
    end
    DateType(Dates.Date(join(fields, "-")), length(fields))
end

function to_lengthtype(x)
    if ismissing(x)
        return missing
    end
    x = string(x)
    plus = false
    if endswith(x, "+")
        x = x[1:end-1]
        plus = true
    end
    LengthType(parse(Float64, x), plus)
end

function to_set(x)
    if ismissing(x)
        return missing
    end
    x = JSON3.read(x)
    if isempty(x)
        return missing
    end
    Set(lowercase.(x))
end

function get_media(source, medium)
    df = CSV.read("$datadir/$(source)_$(medium).csv", DataFrames.DataFrame, ntasks = 1)
    for k in ["startdate", "enddate"]
        tmp = to_datetype.(df[:, k])
        DataFrames.select!(df, DataFrames.Not(k))
        df[:, k] = tmp
    end
    for k in ["episodes", "duration", "volumes", "chapters"]
        tmp = to_lengthtype.(df[:, k])
        DataFrames.select!(df, DataFrames.Not(k))
        df[:, k] = tmp
    end
    for k in ["studios"]
        tmp = to_set.(df[:, k])
        DataFrames.select!(df, DataFrames.Not(k))
        df[:, k] = tmp
    end
    # filter to unique alttitles
    df[:, "titles"] =
        [to_titletype(x, y) for (x, y) in zip(df.title, df.alternative_titles)]
    alttitles = reduce(vcat, [x.alttitles for x in df.titles if !ismissing(x)])
    duplicates = [k for (k, v) in StatsBase.countmap(alttitles) if v > 1]
    for t in df.titles
        if !ismissing(t)
            filter!(x -> x ∉ duplicates, t.alttitles)
        end
    end
    DataFrames.select!(df, DataFrames.Not(["title", "alternative_titles"]))
    df
end

@memoize function get_mediatypes(medium)
    if medium == "manga"
        manga_types = Set(["Manhwa", "Manhua", "Manga", "OEL", "Doujinshi", "One-shot"])
        novel_types = Set(["Light Novel", "Novel"])
        return (manga_types, novel_types)
    elseif medium == "anime"
        tv_types = Set(["ONA", "TV"])
        shortanime_types = Set(["Music", "CM", "PV", "Special"])
        special_types = Set(["OVA", "Special", "TV Special"])
        movie_types = Set(["Movie"])
        return (tv_types, shortanime_types, special_types, movie_types)
    else
        @assert false
    end
end

function match_mediatype(
    medium::String,
    t1::Union{AbstractString,Missing},
    t2::Union{AbstractString,Missing},
    fuzzy::Bool,
)
    if ismissing(t1) || ismissing(t2)
        return 0
    end
    if fuzzy
        for types in get_mediatypes(medium)
            if t1 in types && t2 in types
                return 1
            end
        end
        return -1
    else
        return t1 == t2
    end
end

function match_date(d1::Union{DateType,Missing}, d2::Union{DateType,Missing}, fuzzy::Bool)
    if ismissing(d1) || ismissing(d2)
        return 0
    end
    if fuzzy
        if abs(d1.date - d2.date) <= Dates.Day(31)
            return 1
        end
    end
    N = min(d1.resolution, d2.resolution)
    fns = (Dates.year, Dates.month, Dates.day)
    for i = 1:N
        if fns[i](d1.date) != fns[i](d2.date)
            return fuzzy ? -1 : 0
        end
    end
    1
end

function match_season(
    s1::Union{AbstractString,Missing},
    s2::Union{AbstractString,Missing},
    fuzzy::Bool,
)
    if ismissing(s1) || ismissing(s2)
        return 0
    end
    if s1 == s2
        return 1
    else
        return fuzzy ? -1 : 0
    end
end

@memoize function get_statustypes()
    released = Set(["Finished", "Releasing", "Cancelled", "On Hiatus"])
    unreleased = Set(["TBA", "Upcoming"])
    (released, unreleased)
end

function match_status(
    s1::Union{AbstractString,Missing},
    s2::Union{AbstractString,Missing},
    fuzzy::Bool,
)
    if ismissing(s1) || ismissing(s2)
        return 0
    end
    if fuzzy
        # an item can transition from upcoming -> releasing
        if Set((s1, s2)) == Set(("Upcoming", "Releasing"))
            return 0
        end
        for types in get_statustypes()
            if s1 in types && s2 in types
                return 1
            end
        end
        return -1
    else
        return s1 == s2
    end
end

function match_length(
    N::Int,
    e1::Union{LengthType,Missing},
    e2::Union{LengthType,Missing},
    fuzzy::Bool,
)
    if ismissing(e1) || ismissing(e2)
        return 0
    end
    if fuzzy
        n1 = e1.length
        n2 = e2.length
        if abs(n1 - n2) <= N
            return 1
        elseif min(n1 / n2, n2 / n1) >= 0.8
            return 1
        elseif e1.plus || e2.plus
            return 0
        else
            return -1
        end
    else
        return e1.length == e2.length
    end
end
match_count(e1::Union{LengthType,Missing}, e2::Union{LengthType,Missing}, fuzzy::Bool) =
    match_length(1, e1, e2, fuzzy)
match_duration(e1::Union{LengthType,Missing}, e2::Union{LengthType,Missing}, fuzzy::Bool) =
    match_length(180, e1, e2, fuzzy)

function match_studios(
    s1::Union{Set{String},Missing},
    s2::Union{Set{String},Missing},
    fuzzy::Bool,
)
    if ismissing(s1) || ismissing(s2)
        return 0
    end
    if length(s1) == 0 || length(s2) == 0
        return 0
    end
    if fuzzy
        return !isdisjoint(s1, s2)
    else
        return s1 == s2
    end
end

function matchstring(
    x::Union{AbstractString,Missing},
    y::Union{AbstractString,Missing},
    fuzzy::Bool,
)
    if ismissing(x) || ismissing(y)
        return 0
    end
    if isempty(x) || isempty(y)
        return 0
    end
    if fuzzy
        cutoff = 0.9
        match = StringDistances.compare(x, y, StringDistances.Levenshtein()) > cutoff
        return match ? 1 : -1
    else
        return x == y
    end
end

macro earlyreturn(errcode, accum, expr)
    esc(quote
        _r = $(expr)
        if _r == $(errcode)
            return $(errcode)
        else
            $(accum) += _r
        end
    end)
end

function match_titles(
    t1::Union{TitleType,Missing},
    t2::Union{TitleType,Missing},
    fuzzy::Bool,
)
    if ismissing(t1) || ismissing(t2)
        return 0
    end
    n = 0
    @earlyreturn 1 n matchstring(t1.title, t2.title, fuzzy)
    for x in union(Set([t1.title]), Set(t1.alttitles))
        for y in union(Set([t2.title]), Set(t2.alttitles))
            @earlyreturn 1 n matchstring(x, y, fuzzy)
        end
    end
    fuzzy ? -1 : 0
end

function fuzzy(fn::Function, args...)
    n = 0
    @earlyreturn -1 n fn(args..., true)
    @earlyreturn -1 n fn(args..., false)
    n
end

function match_rows(medium, df1, i, df2, j)
    n = 0
    @earlyreturn -1 n fuzzy(match_mediatype, medium, df1.mediatype[i], df2.mediatype[j])
    @earlyreturn -1 n fuzzy(match_date, df1.startdate[i], df2.startdate[j])
    @earlyreturn -1 n fuzzy(match_date, df1.enddate[i], df2.enddate[j])
    @earlyreturn -1 n fuzzy(match_season, df1.season[i], df2.season[j])
    @earlyreturn -1 n fuzzy(match_status, df1.status[i], df2.status[j])
    @earlyreturn -1 n fuzzy(match_count, df1.episodes[i], df2.episodes[j])
    @earlyreturn -1 n fuzzy(match_count, df1.chapters[i], df2.chapters[j])
    @earlyreturn -1 n fuzzy(match_count, df1.volumes[i], df2.volumes[j])
    @earlyreturn -1 n fuzzy(match_duration, df1.duration[i], df2.duration[j])
    @earlyreturn -1 n fuzzy(match_studios, df1.studios[i], df2.studios[j])
    @earlyreturn -1 n fuzzy(match_titles, df1.titles[i], df2.titles[j])
    n
end

function match_metadata(source1, source2, medium, idxs, showprogress)
    media1 = get_media(source1, medium)
    media2 = get_media(source2, medium)
    matches = Dict()
    @showprogress enabled = showprogress for i in idxs
        candidate = nothing
        max_matches = 0
        for j = 1:DataFrames.nrow(media2)
            nmatches = match_rows(medium, media1, i, media2, j)
            if nmatches == max_matches
                candidate = nothing
            elseif nmatches > max_matches
                candidate = j
                max_matches = nmatches
            end
        end
        if !isnothing(candidate)
            matches[media1.itemid[i]] = media2.itemid[candidate]
        end
    end
    matches
end

function match_metadata(source1, source2, medium)
    nchunks = Threads.nthreads()
    idxs = Random.shuffle(1:DataFrames.nrow(get_media(source1, medium)))
    chunks = Iterators.partition(idxs, div(length(idxs), nchunks))
    tasks = map(Iterators.enumerate(chunks)) do (i, chunk)
        Threads.@spawn @handle_errors match_metadata(source1, source2, medium, chunk, i == 1)
    end
    matches = fetch.(tasks)
    reduce(merge, matches)
end

function save_matches()
    outdir = "$datadir/metadata"
    mkpath(outdir)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    for medium in ["manga", "anime"]
        for i = 1:length(sources)
            for j = i+1:length(sources)
                matches = match_metadata(sources[j], sources[i], medium)
                mappings = [(k, v) for (k, v) in matches]
                colnames = ["source1", "source2"]
                if isempty(mappings)
                    df = DataFrames.DataFrame(Dict(x => [] for x in colnames))
                else
                    df = DataFrames.DataFrame(mappings, colnames)
                end
                CSV.write("$outdir/$medium.$(sources[j]).$(sources[i]).csv", df)
            end
        end
    end
end

save_matches()
