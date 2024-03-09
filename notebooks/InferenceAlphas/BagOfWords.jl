import HDF5
import HTTP
import JSON
import NBInclude: @nbinclude
import Memoize: @memoize
import Oxygen
import SparseArrays: AbstractSparseArray, sparse
@nbinclude("../TrainingAlphas/Alpha.ipynb")
@nbinclude("../TrainingAlphas/Baseline/BaselineHelper.ipynb")

# Baseline

num_threads() = 1
get_training_counts(df, col) = get_counts(getfield(df, col))

function get_item_weights(medium, df, λ, counts)
    w = zeros(Float32, length(df.itemid))
    for i = 1:length(df.itemid)
        a = df.itemid[i]
        if a ∈ keys(counts)
            w[i] = counts[a]
        end
    end
    powerdecay(w, log(λ))
end

function get_weights(medium, df, λ_wu, λ_wa, λ_wt, item_counts)
    user_weight = powerdecay(get_training_counts(df, :userid), log(λ_wu))
    item_weight = get_item_weights(medium, df, λ_wa, item_counts)
    timestamp_weight = λ_wt .^ (1 .- df.updated_at)
    user_weight .* item_weight .* timestamp_weight
end

function train_model(medium, training, λ, μ, a, item_counts)
    if length(training.rating) == 0
        return μ
    end
    λ_u, _, λ_wu, λ_wa, λ_wt = λ
    users, items, ratings = training.userid, training.itemid, training.rating
    weights = get_weights(medium, training, λ_wu, λ_wa, λ_wt, item_counts)
    u = zeros(eltype(λ_u), 1)
    ρ_u = zeros(eltype(u), length(u), Threads.nthreads())
    Ω_u = zeros(eltype(u), length(u), Threads.nthreads())
    update_users!(users, items, ratings, weights, u, a, λ_u, ρ_u, Ω_u; μ = μ)
    u
end

@memoize read_params_memoized(args...; kwargs...) = read_params(args...; kwargs...)

function compute_baseline_alpha(username, source, medium)
    training = get_split(
        "rec_training",
        "rating",
        medium,
        [:userid, :itemid, :rating, :update_order, :updated_at],
        nothing,
        username,
        source,
    )
    alpha = "$medium/Baseline/rating"
    params = read_params_memoized(alpha, false)
    u = train_model(
        medium,
        training,
        params["λ"],
        mean(params["u"]),
        params["a"],
        params["item_counts"],
    )
    preds = make_prediction(
        fill(0, num_items(medium)),
        collect(0:num_items(medium)-1),
        u,
        params["a"],
    )
    model(userids, itemids) = [preds[x+1] for x in itemids]
    write_alpha(model, medium, alpha, REC_SPLITS, username, source)
end

# BagOfWords

@memoize function get_rating_beta(name)
    params = read_params(name, false)
    params["β"]
end

function get_inputs(medium::String, metric::String, username::String, source::String)
    split = "rec_training"
    fields = [:userid, :itemid, :metric]
    if metric == "rating"
        alpha = "$medium/Baseline/rating"
        β = get_rating_beta(alpha)
        df = get_split(split, metric, medium, fields, alpha, username, source)
        df.metric .= df.metric - df.alpha .* β
    else
        df = get_split(split, metric, medium, fields, nothing, username, source)
    end
    sparse(df, medium)
end

function get_epoch_inputs(username::String, source::String)
    GC.enable(false)
    inputs = [
        get_inputs(medium, metric, username, source) for metric in ["rating", "watch"]
        for medium in ALL_MEDIUMS
    ]
    X = vcat(inputs...)
    GC.enable(true)
    X
end

function save_features(outdir::String, username::String, source::String)
    mkpath(outdir)
    filename = joinpath(outdir, "inference.h5")
    d = Dict{String,Any}()
    X = get_epoch_inputs(username, source)
    record_sparse_array!(d, "inputs", X)
    d["epoch_size"] = 1
    d["users"] = [0]
    HDF5.h5open(filename, "w") do file
        for (k, v) in d
            file[k] = v
        end
    end
end

function record_sparse_array!(d::Dict, name::String, x::AbstractSparseArray)
    i, j, v = SparseArrays.findnz(x)
    d[name*"_i"] = i
    d[name*"_j"] = j
    d[name*"_v"] = v
    d[name*"_size"] = [size(x)[1], size(x)[2]]
end

function compute_bagofwords_alpha(username, source, medium, metric, version, modelport)
    cmdstring = (
        "curl http://localhost:$modelport/query?username=$username&source=$source&medium=$medium&metric=$metric"
    )
    cmd = Cmd(String.(split(cmdstring)))
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    outdir = joinpath(
        get_data_path("recommendations/$source/$username"),
        "alphas",
        medium,
        "BagOfWords",
        version,
        metric,
    )
    file = HDF5.h5open(joinpath(outdir, "predictions.h5"), "r")
    e = vec(read(file["predictions"]))
    if metric in ["watch", "plantowatch"]
        # make preds for regular items
        seen =
            get_raw_split(
                "rec_training",
                medium,
                [:itemid],
                nothing,
                username,
                source,
            ).itemid
        r = copy(e)
        r[seen.+1] .= 0
        r = r ./ sum(r)
        # make preds for plantowatch items
        ptw =
            get_split(
                "rec_training",
                "plantowatch",
                medium,
                [:itemid],
                nothing,
                username,
                source,
            ).itemid
        p = copy(e)
        watched = setdiff(Set(seen), Set(ptw))
        p[watched.+1] .= 0
        p = p ./ sum(p)
        # combine preds
        e = copy(r)
        e[ptw.+1] .= p[ptw.+1]
    elseif metric in ["rating", "drop"]
        nothing
    else
        @assert false
    end
    close(file)
    model(userids, itemids) = [e[x+1] for x in itemids]
    write_alpha(
        model,
        medium,
        "$medium/BagOfWords/$version/$metric",
        REC_SPLITS,
        username,
        source,
    )
end

Oxygen.@get "/query" function (req::HTTP.Request)
    params = Oxygen.queryparams(req)
    username = params["username"]
    source = params["source"]
    modelport = params["modelport"]
    version = "v1"
    outdir = joinpath(
        get_data_path("recommendations/$source/$username"),
        "alphas",
        "BagOfWords",
        version,
    )
    for medium in ALL_MEDIUMS
        compute_baseline_alpha(username, source, medium)
    end    
    save_features(outdir, username, source)
    @sync for medium in ALL_MEDIUMS
        for metric in ALL_METRICS
            Threads.@spawn compute_bagofwords_alpha(username, source, medium, metric, version, modelport)
        end
    end
end

Oxygen.serveparallel(; port=parse(Int, ARGS[1]))