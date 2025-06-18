import CSV
import DataFrames
import JLD2
import HDF5
import LinearAlgebra

function get_embedding_matrix(medium)
    registry = Dict()
    HDF5.h5open("../../../data/finetune/clip/output.embeddings.$medium.h5", "r") do f
        for k in keys(f)
            v = read(f, k)
            v = convert.(Float32, v)
            registry[k] = convert.(Float32, v)
        end
    end
    registry
    M = zeros(Float32, length(first(values(registry))), length(registry))
    for (k, v) in registry
        c = parse(Int, split(k, ".")[end])
        M[:, c+1] = v
    end
    M
end

function get_adaptations(split, medium)
    df = CSV.read("../../../data/finetune/clip/$split.csv", DataFrames.DataFrame)
    DataFrames.filter!(x -> x.cliptype == "adaptation$medium", df)
    source_ids = Int[]
    target_ids = Int[]
    for i = 1:DataFrames.nrow(df)
        append!(source_ids, df.source_matchedid[i])
        append!(target_ids, df.target_matchedid[i])
    end
    source_ids, target_ids
end

function closest_orthogonal_map(A, B)
    M = B * A'
    x = LinearAlgebra.svd(M)
    x.U * x.Vt
end

function avg_norm(x)
    sum(x .^ 2) / (size(x)[1] * size(x)[2])
end

function get_cross_medium_map(d, medium)
    A = d["embeddings.$medium"]
    B = d["embeddings.$(1-medium)"]
    sourceids, targetids = get_adaptations("training", medium)
    M = closest_orthogonal_map(A[:, sourceids.+1], B[:, targetids.+1])
end


function save_clip()
    d = Dict("embeddings.$m" => get_embedding_matrix(m) for m in [0, 1])
    for medium in [0, 1]
        A = d["embeddings.$medium"]
        B = d["embeddings.$(1-medium)"]
        s_training, t_training = get_adaptations("training", medium)
        M = closest_orthogonal_map(A[:, s_training.+1], B[:, t_training.+1])
        training_loss = avg_norm(M * A[:, s_training.+1] - B[:, t_training.+1])
        s_test, t_test = get_adaptations("test", medium)
        test_loss = avg_norm(M * A[:, s_test.+1] - B[:, t_test.+1])
        println("medium $medium, training loss $training_loss, test loss $test_loss")
        d["crossproject.$medium"] = M
    end
    JLD2.save("../../../data/finetune/clip.jld2", d)
end

save_clip()
