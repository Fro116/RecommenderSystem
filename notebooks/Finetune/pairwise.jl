import CSV
import DataFrames
import JLD2
import HDF5
import LinearAlgebra
import ProgressMeter: @showprogress, next!

const datadir = "../../data/finetune"

function get_adaptations(medium::Int)
    df = CSV.read("$datadir/media_relations.csv", DataFrames.DataFrame)
    filter!(
        x ->
            x.source_medium == medium &&
                x.target_medium != medium &&
                x.relation in ["adaptation", "source"],
        df,
    )
    df = DataFrames.DataFrame(
        cliptype = "adaptation$medium",
        source_matchedid = df.source_matchedid,
        target_matchedid = df.target_matchedid,
    )
    df = DataFrames.filter(x -> x.source_matchedid != 0 && x.target_matchedid != 0, df)
    df
end

function save_adaptations(test_frac::Float64)
    df = reduce(vcat, [get_adaptations.([0, 1])...])
    function reflect(x)
        cliptype, sourceid, targetid = x
        reflect_type = Dict("adaptation0" => "adaptation1", "adaptation1" => "adaptation0")
        (reflect_type[cliptype], targetid, sourceid)
    end
    ids = Set()
    for i = 1:DataFrames.nrow(df)
        k = (df.cliptype[i], df.source_matchedid[i], df.target_matchedid[i])
        push!(ids, min(k, reflect(k)))
    end
    test_ids = Set(x for x in ids if rand() < test_frac)
    test_ids = union(test_ids, Set(reflect.(test_ids)))
    test_mask = [
        (x.cliptype, x.source_matchedid, x.target_matchedid) in test_ids for
        x in eachrow(df)
    ]
    training_df = df[.!test_mask, :]
    test_df = df[test_mask, :]
    CSV.write("$datadir/adaptations.training.csv", training_df)
    CSV.write("$datadir/adaptations.test.csv", test_df)
end

function get_embedding_matrix(medium)
    JLD2.load("$datadir/pairwise.embeddings.jld2")["embeddings.$medium"]
end

function get_adaptations(split, medium)
    df = CSV.read("$datadir/adaptations.$split.csv", DataFrames.DataFrame)
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

function make_metric_dataframe(dict)
    function natural_sort_key(s::AbstractString)
        parts = []
        pattern = r"([^\d]+)|(\d+)"
        for m in eachmatch(pattern, s)
            if m.captures[1] === nothing
                push!(parts, parse(Int, m.match))
            else
                push!(parts, m.match)
            end
        end
        return parts
    end
    result = Dict{Int,Dict{Symbol,Any}}()
    for (k, v) in dict
        task, metric = match(r"^(\d+)\.(.+)$", k).captures
        task_id = parse(Int, task)
        result[task_id] = get(result, task_id, Dict{Symbol,Any}())
        result[task_id][Symbol(metric)] = v
    end
    df = DataFrames.DataFrame([merge(Dict(:medium => k), v) for (k, v) in result])
    other_cols = filter(!=(Symbol("medium")), names(df))
    sorted_cols = sort(other_cols, by = col -> natural_sort_key(String(col)))
    df = df[:, DataFrames.Cols(:medium, sorted_cols...)]
    for col in names(df)
        if all(x -> x isa AbstractVector, df[!, col])
            df[!, col] = JSON3.write.(df[!, col])
        end
    end
    df
end

function save_item_similarity_model()
    save_adaptations(0.01)
    metrics = Dict()
    d = Dict("embeddings.$m" => get_embedding_matrix(m) for m in [0, 1])
    for medium in [0, 1]
        A = d["embeddings.$medium"]
        B = d["embeddings.$(1-medium)"]
        s_training, t_training = get_adaptations("training", medium)
        M = closest_orthogonal_map(A[:, s_training.+1], B[:, t_training.+1])
        training_loss = avg_norm(M * A[:, s_training.+1] - B[:, t_training.+1])
        s_test, t_test = get_adaptations("test", medium)
        test_loss = avg_norm(M * A[:, s_test.+1] - B[:, t_test.+1])
        metrics["$medium.project.training"] = training_loss
        metrics["$medium.project.test"] = test_loss
        d["crossproject.$medium"] = M
    end
    JLD2.save("$datadir/item_similarity.jld2", d)
    metrics = make_metric_dataframe(metrics)
    display(metrics)
    CSV.write("$datadir/item_similarity.csv", metrics)
end

save_item_similarity_model()
