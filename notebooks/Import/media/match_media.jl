import DataFrames
import ProgressMeter: @showprogress
import StatsBase
include("../../julia_utils/stdout.jl")
include("common.jl")

const itemtype = Tuple{String,Union{AbstractString,Int}}
const edgetype = Tuple{itemtype,itemtype}

@memoize function get_items(medium::String, source::String)
    df = read_csv("$datadir/users/$source/$medium.csv")
    Dict(df.itemid .=> df.count)
end

function get_edges(edgetype::String, medium::String, source1::String, source2::String)
    df = read_csv("$datadir/$edgetype/$medium.$source1.$source2.csv")
    items1 = get_items(medium, source1)
    items2 = get_items(medium, source2)
    edges::Set{Tuple{itemtype,itemtype}} = Set()
    for (u, v) in zip(df.source1, df.source2)
        if u in keys(items1) && v in keys(items2)
            push!(edges, ((source1, u), (source2, v)))
        end
    end
    edges
end

str(x::itemtype, y::itemtype) = join([x..., y...], ",")
str(x::edgetype) = str(x...)

function get_vertices(medium::String, source::String)
    v = Dict((source, k) => Set([(source, k)]) for (k, _) in get_items(medium, source))
    for (x, y) in get_edges("merge", medium, source, source)
        if v[x] == v[y]
            logerror("get_vertices: stale edge $(str(x, y))")
            continue
        end
        z = union(v[x], v[y])
        for w in z
            v[w] = z
        end
    end
    v
end

function get_edges(
    edgetypes::Vector{String},
    medium::String,
    source1::String,
    source2::String,
)
    vs = get_vertices(medium, source2)
    edges = Dict()
    invalid = get_edges("invalid", medium, source1, source2)
    valid = get_edges("valid", medium, source1, source2)
    mismarked = Set()
    for edgetype in edgetypes
        for edge in get_edges(edgetype, medium, source1, source2)
            if edge in invalid
                push!(mismarked, edge)
                continue
            end
            if edgetype != "valid" && edge in valid
                logerror("get_edges: stale valid $(str(edge))")
            end
            conflicts = false
            for e in get(edges, edge[1], [])
                if vs[edge[end]] != vs[e[end]]
                    conflicts = true
                    logerror("get_edges: $(str(e)) conflicts $(str(edge))")
                    break
                end
            end
            if conflicts
                continue
            end
            if edge[1] ∉ keys(edges)
                edges[edge[1]] = Set()
            end
            push!(edges[edge[1]], edge)
        end
    end
    for x in setdiff(invalid, mismarked)
        logerror("get_edges: stale invalid $(str(x))")
    end
    reduce(union, collect(values(edges)))
end

function get_graph(sources::Vector{String}, edgetypes::Vector{String}, medium::String)
    vs = reduce(merge, get_vertices.(medium, sources))
    es = []
    get_sources(v) = Set(x for (x, _) in v)
    for i = 1:length(sources)
        for j = i+1:length(sources)
            @showprogress for (x, y) in get_edges(edgetypes, medium, sources[j], sources[i])
                if vs[x] == vs[y]
                    continue
                end
                if length(get_sources(vs[x])) != 1
                    logerror(
                        "get_graph: $(string(vs[x])) conflicts $(string(vs[y])) | $(str((x,y)))",
                    )
                    continue

                end
                z = union(vs[x], vs[y])
                for w in z
                    vs[w] = z
                end
                push!(es, (x, y))
            end
        end
    end
    vs
end

function save_graph(vs, medium::String)
    records = []
    id_map = Dict()
    for (k, v) in vs
        if v ∉ keys(id_map)
            id_map[v] = length(id_map) + 1
        end
        push!(records, (k..., id_map[v], get_items(medium, k[1])[k[2]]))
    end
    df = DataFrames.DataFrame(records, ["source", "itemid", "groupid", "count"])
    write_csv("$datadir/$medium.groups.csv", df)
end

function match_media()
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    edgetypes = ["valid", "manami", "ids", "metadata"]
    for m in ["manga", "anime"]
        logtag("MATCH_MEDIA", "matching $m")
        vs = get_graph(sources, edgetypes, m)
        save_graph(vs, m)
    end
end

match_media()
