import NBInclude: @nbinclude    
if !@isdefined EXPLICIT_IFNDEF
    EXPLICIT_IFNDEF = true        
    source_name = "ExplicitUserItemBiases"
    
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Explicit/ExplicitUserItemBiasesBase.ipynb")

    function train_model(training, λ, μ, a)
        if length(training.rating) == 0
            return μ
        end
        λ_u, _, λ_wu, λ_wa, λ_wt = λ
        users, items, ratings = training.user, training.item, training.rating
        weights =
            powerdecay(length(training.item), log(λ_wu)) .* powerdecay(
                get_counts(
                    "training",
                    "all",
                    "explicit",
                    training.medium;
                    by_item = true,
                    per_rating = false,
                )[training.item],
                log(λ_wa),
            ) .* powerlawdecay(1 .- training.timestamp, λ_wt)
        u = zeros(eltype(λ_u), maximum(users))
        ρ_u = zeros(eltype(u), length(u), Threads.nthreads())
        Ω_u = zeros(eltype(u), length(u), Threads.nthreads())
        update_users!(users, items, ratings, weights, u, a, λ_u, ρ_u, Ω_u; μ = μ)
        u
    end

    function compute_alpha(task, medium)
        training = get_recommendee_split("explicit", medium)
        params = read_params("$medium/$task/$source_name")
        u = train_model(training, params["λ"], mean(params["u"]), params["a"])
        preds = make_prediction(fill(1, num_items(medium)), collect(1:num_items(medium)), u, params["a"])
        write_recommendee_alpha(preds, medium, "$medium/$task/$source_name")
    end
end
    
username = ARGS[1]
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_alpha(task, medium)
    end
end