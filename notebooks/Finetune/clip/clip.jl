import CSV
import DataFrames
import JLD2
import HDF5
import LinearAlgebra

const datadir = "../../../data/finetune"

function get_embedding_matrix(medium)
    registry = Dict()
    HDF5.h5open("$datadir/clip/output.embeddings.$medium.h5", "r") do f
        for k in keys(f)
            if !startswith(k, "$medium")
                continue
            end
            v = read(f, k)
            v = convert.(Float32, v)
            registry[k] = convert.(Float32, v)
        end
    end
    M = zeros(Float32, length(first(values(registry))), length(registry))
    for (k, v) in registry
        c = parse(Int, split(k, ".")[end])
        M[:, c+1] = v
    end
    M
end

function get_adaptations(split, medium)
    df = CSV.read("$datadir/clip/$split.csv", DataFrames.DataFrame)
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

function dcg_at_k(relevances::Vector{<:Real}, k::Int)
    len = min(k, length(relevances))
    if len == 0
        return 0.0
    end
    score = 0.0
    for i = 1:len
        score += relevances[i] / log2(i + 1)
    end
    score
end

function ndcg_at_k(df::DataFrames.DataFrame, M::Matrix{<:Real}, k::Int)
    unique_sources = unique(df.source)
    weighted_ndcgs = Vector{Float64}(undef, length(unique_sources))
    source_weights = Vector{Float64}(undef, length(unique_sources))
    num_items = size(M, 2)
    Threads.@threads for i = 1:length(unique_sources)
        source_id = unique_sources[i]
        ground_truth_pairs = filter(row -> row.source == source_id, df)
        weight = first(ground_truth_pairs.popularity)
        true_relevances =
            Dict(pair.target => pair.relevance for pair in eachrow(ground_truth_pairs))
        predictions = M[source_id, :]
        candidate_items = filter(item -> item != source_id, 1:num_items)
        ranked_item_indices = sortperm(predictions[candidate_items], rev = true)
        ranked_items = candidate_items[ranked_item_indices]
        ranked_relevances = [get(true_relevances, item_id, 0.0) for item_id in ranked_items]
        dcg = dcg_at_k(ranked_relevances, k)
        ideal_relevances = sort(collect(values(true_relevances)), rev = true)
        idcg = dcg_at_k(ideal_relevances, k)
        ndcg = idcg > 0 ? dcg / idcg : 0.0
        weighted_ndcgs[i] = ndcg * weight
        source_weights[i] = weight
    end
    sum(weighted_ndcgs) / sum(source_weights)
end

function recall_at_k(df::DataFrames.DataFrame, M::Matrix{<:Real}, k::Int)
    unique_sources = unique(df.source)
    weighted_recalls = Vector{Float64}(undef, length(unique_sources))
    source_weights = Vector{Float64}(undef, length(unique_sources))
    num_items = size(M, 2)
    Threads.@threads for i = 1:length(unique_sources)
        source_id = unique_sources[i]
        source_rows = filter(row -> row.source == source_id, df)
        predictions = M[source_id, :]
        candidate_items = filter(item -> item != source_id, 1:num_items)
        ranked_item_indices = sortperm(predictions[candidate_items], rev = true)
        top_k_range = 1:min(k, length(ranked_item_indices))
        top_k_items = candidate_items[ranked_item_indices[top_k_range]]
        ground_truth_counts = Dict(row.target => row.count for row in eachrow(source_rows))
        count_in_top_k =
            sum(get(ground_truth_counts, item_id, 0) for item_id in top_k_items)
        recall = count_in_top_k / sum(source_rows.count)
        weight = first(source_rows.popularity)
        weighted_recalls[i] = recall * weight
        source_weights[i] = weight
    end
    sum(weighted_recalls) / sum(source_weights)
end

function get_ranking_metrics()
    ret = Dict()
    for medium in [0, 1]
        d = JLD2.load("$datadir/clip.jld2")
        train_df = CSV.read("$datadir/clip/training.csv", DataFrames.DataFrame)
        train_ids = Set([(x.cliptype, x.source_matchedid) for x in eachrow(train_df)])
        test_df = CSV.read("$datadir/clip/test.csv", DataFrames.DataFrame)
        test_mask = [(x.cliptype, x.source_matchedid) ∉ train_ids for x in eachrow(test_df)]
        df = test_df[test_mask, :]
        df = filter(x -> x.cliptype == "medium$medium" && x.count > 0, df)
        df = DataFrames.DataFrame(
            source = df.source_matchedid .+ 1,
            target = df.target_matchedid .+ 1,
            relevance = 1 .+ log10.(df.count),
            count = df.count,
            popularity = 1,
        )
        M = d["embeddings.$medium"]' * d["embeddings.$medium"]
        for k in [8, 128, 1024]
            ret["$medium.nDCG@$k"] = ndcg_at_k(df, M, k)
            ret["$medium.Recall@$k"] = recall_at_k(df, M, k)
        end
    end
    ret
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

function save_clip()
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
    JLD2.save("$datadir/clip.jld2", d)
    metrics = make_metric_dataframe(merge(get_ranking_metrics(), metrics))
    display(metrics)
    CSV.write("$datadir/clip.csv", metrics)
end

save_clip()
