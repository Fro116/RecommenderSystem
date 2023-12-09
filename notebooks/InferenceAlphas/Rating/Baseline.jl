import NBInclude: @nbinclude
if !@isdefined BASELINE_IFNDEF
    BASELINE_IFNDEF = true
    source_name = "ExplicitUserItemBiases"
    @nbinclude("../../TrainingAlphas/Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Rating/BaselineHelper.ipynb")

    function train_model(training, λ, μ, a)
        if length(training.rating) == 0
            return μ
        end
        λ_u, _, λ_wu, λ_wa, λ_wt = λ
        users, items, ratings = training.userid, training.itemid, training.rating
        weights = get_weights(λ_wu, λ_wa, λ_wt)
        u = zeros(eltype(λ_u), 1)
        ρ_u = zeros(eltype(u), length(u), Threads.nthreads())
        Ω_u = zeros(eltype(u), length(u), Threads.nthreads())
        update_users!(users, items, ratings, weights, u, a, λ_u, ρ_u, Ω_u; μ = μ)
        u
    end

    function compute_alpha(medium)
        training = get_split(
            "rec_training",
            "rating",
            medium,
            [:userid, :itemid, :rating, :update_order, :updated_at],
        )
        alpha = "$medium/rating/Baseline"
        params = read_params(alpha)
        u = train_model(training, params["λ"], mean(params["u"]), params["a"])
        preds = make_prediction(
            fill(0, num_items(medium)),
            collect(0:num_items(medium)-1),
            u,
            params["a"],
        )
        model(userids, itemids) = [preds[x+1] for x in itemids]
        write_alpha(model, medium, alpha, REC_SPLITS)
    end
end

username = ARGS[1]
for medium in ALL_MEDIUMS
    compute_alpha(medium)
end