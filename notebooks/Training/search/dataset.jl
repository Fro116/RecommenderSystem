import CSV
import DataFrames
import H5Zblosc
import HDF5
import JLD2
import JSON3
import Memoize: @memoize
import Random
import SparseArrays
include("../../julia_utils/stdout.jl")
include("../../Import/embeddings/queries.jl")

const datadir = "../../../data/training"
const secretdir = "../../../secrets"

@memoize function num_items(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

function get_query_dataset(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = open("$datadir/$m.json") do f
        JSON3.read(f)
    end
    queries = []
    for x in df
        for (query, tag) in get_queries(x)
            push!(queries, (x[:matchedid], query, tag))
        end
    end
    queries
end

# TODO graded tags

function save_split(medium::Int, name::String, data::Vector, embeddings)
    chunk_embeddings = []
    chunk_targets = []
    chunk_weights = []
    for (k, v) in data
        if k ∉ keys(embeddings)
            logerror("could not find embedding for query $k")
            continue
        end
        targets = SparseArrays.spzeros(Float32, num_items(medium))
        for (itemid, tag) in v
            targets[itemid+1] = 1
        end
        targets ./= sum(targets)
        push!(chunk_embeddings, embeddings[k])
        push!(chunk_targets, targets)
        push!(chunk_weights, Float32[1])
    end
    chunk_embeddings_mat = reduce(hcat, chunk_embeddings)
    chunk_targets_mat = reduce(hcat, chunk_targets)
    chunk_weights_mat = reduce(hcat, chunk_weights)
    HDF5.h5open("$datadir/search/$name.$medium.h5", "w") do f
        f["query_embeddings", blosc = 3] = chunk_embeddings_mat
        f["targets_colptr", blosc = 3] = chunk_targets_mat.colptr
        f["targets_rowval", blosc = 3] = chunk_targets_mat.rowval
        f["targets_nzval", blosc = 3] = chunk_targets_mat.nzval
        f["targets_size"] = collect(size(chunk_targets_mat))
        f["weights", blosc = 3] = chunk_weights_mat
    end
end

function save_medium(medium::Int, embeddings)
    df = get_query_dataset(medium)
    d = Dict()
    for x in df
        itemid, query, tag = x
        if query ∉ keys(d)
            d[query] = []
        end
        push!(d[query], (itemid, tag))
    end
    d = Random.shuffle(collect(d))
    train_perc = 0.9
    train_test_cutoff = Int(round(length(d) * train_perc))
    save_split(medium, "training", d[1:train_test_cutoff], embeddings)
    save_split(medium, "test", d[train_test_cutoff+1:end], embeddings)
end

function save_dataset()
    rm("$datadir/search", force = true, recursive = true)
    mkdir("$datadir/search")
    embeddings = JLD2.load("$datadir/query_embeddings.jld2")["queries"]
    for m in [0, 1]
        save_medium(m, embeddings)
    end
end

save_dataset()
