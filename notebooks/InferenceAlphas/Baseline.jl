import HTTP
import NBInclude: @nbinclude
import Memoize: @memoize
import Oxygen

@nbinclude("../TrainingAlphas/Alpha.ipynb")
@nbinclude("../TrainingAlphas/Baseline/BaselineHelper.ipynb")

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

function compute_alpha(username, source, medium)
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

Oxygen.@get "/query" function (req::HTTP.Request)
    params = Oxygen.queryparams(req)
    username = params["username"]
    source = params["source"]
    for medium in ALL_MEDIUMS
        compute_alpha(username, source, medium)
    end
end

Oxygen.serveparallel(; port=parse(Int, ARGS[1]))