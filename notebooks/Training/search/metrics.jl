import JLD2
import H5Zblosc
import HDF5

const datadir = "../../../data/training/search"

function get_embedding_matrix(medium)
    HDF5.h5open("$datadir/search.retrieval.embeddings.$medium.h5", "r") do f
        v = read(f, "search.$medium")
        return convert.(Float32, v)
    end
end

function save_metrics()
    ret = Dict()
    d = Dict("embeddings.$m" => get_embedding_matrix(m) for m in [0, 1])
    JLD2.save("$datadir/search.embeddings.jld2", d)
end

save_metrics()