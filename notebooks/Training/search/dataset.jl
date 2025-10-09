import JupyterFormatter
JupyterFormatter.enable_autoformat()

import CSV
import DataFrames
import H5Zblosc
import HDF5
import HTTP
import JLD2
import JSON3
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random
import StatsBase
include("../../julia_utils/stdout.jl")

const datadir = "../../../data/training"
const secretdir = "../../../secrets"

@memoize function num_items(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

function get_view_counts(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    rel = zeros(Float32, num_items(medium))
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame)
    df = DataFrames.combine(DataFrames.groupby(df, [:source, :matchedid])) do subdf
        subdf[argmax(subdf.count), :]
    end
    for x in eachrow(df)
        rel[x["matchedid"]+1] += x["count"]
    end
    rel
end

function extract_keywords(document_text::String)::Vector{String}
    pattern = r"(?s)# Keywords\s*\n(.*?)(?=\n\s*#|\z)"
    m = match(pattern, document_text)
    if m === nothing
        return String[]
    end
    keyword_block = m.captures[1]
    keywords = [strip(kw) for kw in split(keyword_block, ',') if !isempty(strip(kw))]
    keywords
end

function get_date_tags(data)
    tags = []
    if !isnothing(data[:season])
        push!(tags, replace(data[:season], "-" => " "))
    end
    if !isnothing(data[:startdate]) && !isempty(data[:startdate])
        if length(data[:startdate]) >= 4
            push!(tags, data[:startdate][1:4])
        end
        start_year = parse(Int, data[:startdate][1:4])
        decade_start = floor(Int, start_year / 10) * 10
        if decade_start >= 2000
            push!(tags, "$(decade_start)s")
        else
            push!(tags, "$(decade_start % 100)s")
        end
    end
    tags
end

function get_queries(json)
    tags = []
    if !isnothing(json[:llm_summary])
        append!(tags, extract_keywords(json[:llm_summary][:text]))
    end
    append!(tags, copy(json[:genres]))
    append!(tags, copy(json[:tags]))
    ret = [(x, "tag") for x in tags]
    for k in [:title, :english_title]
        if !isnothing(json[k]) && !isempty(json[k])
            push!(ret, (json[k], "title"))
        end
    end
    for x in get_date_tags(json[:metadata][:dates])
        try
            push!(ret, (x, "date"))
        catch
            println((json[:metadata][:dates]))
        end
    end
    collect(Set([(lowercase(x), y) for (x, y) in ret]))
end

function get_queries(medium::Int)
    counts = get_view_counts(medium)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = open("$datadir/$m.json") do f
        JSON3.read(f)
    end
    queries = []
    for x in df
        for (query, tag) in get_queries(x)
            push!(queries, (medium, x[:matchedid], query, tag, counts[x[:matchedid]+1]))
        end
    end
    queries
end

function save_queries(queries, embs, name)
    category_map = Dict("tag" => 0, "title" => 1, "date" => 2)

    chunk_size = 100_000
    for (chunk_idx, chunk) in Iterators.enumerate(Iterators.partition(queries, chunk_size))
        mediums = Int32[]
        matchedids = Int32[]
        qembs = Vector{Float32}[]
        categories = Int32[]
        counts = Float32[]
        for (medium, matchedid, query, category, count) in chunk
            if query ∉ keys(embs)
                logerror("could not find embedding for $query")
                continue
            end
            push!(mediums, medium)
            push!(matchedids, matchedid)
            push!(qembs, embs[query])
            push!(categories, category_map[category])
            push!(counts, count)
        end
        HDF5.h5open("$datadir/search/$name.$chunk_idx.h5", "w") do f
            f["mediums", blosc = 3] = mediums
            f["matchedids", blosc = 3] = matchedids
            f["queries", blosc = 3] = reduce(hcat, qembs)
            f["categories", blosc = 3] = categories
            f["counts", blosc = 3] = counts
        end
    end
end

function save_dataset()
    queries = reduce(vcat, get_queries.([0, 1]))
    Random.shuffle!(queries)
    test_frac = 0.1
    test_cutoff = Int(round(length(queries) * test_frac))
    test_queries = queries[1:test_cutoff]
    train_queries = queries[test_cutoff+1:end]
    mkpath("$datadir/search")
    embs = JLD2.load("$datadir/search_embeddings.jld2")["queries"]
    save_queries(train_queries, embs, "training")
    save_queries(test_queries, embs, "test")
end

save_dataset()
