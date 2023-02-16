#   Explicit
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#     •  See the corresponding file in ../../TrainingAlphas for more
#        details

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
        λ_u, _, λ_wu, λ_wa = λ
        users, items, ratings = training.user, training.item, training.rating
        weights =
            powerdecay(length(training.item), log(λ_wu)) .* powerdecay(
                get_counts("training", "all", "explicit"; by_item = true, per_rating = false)[training.item],
                log(λ_wa),
            )
        u = zeros(eltype(λ_u), maximum(users))
        ρ_u = zeros(eltype(u), length(u), Threads.nthreads())
        Ω_u = zeros(eltype(u), length(u), Threads.nthreads())
        update_users!(users, items, ratings, weights, u, a, λ_u, ρ_u, Ω_u; μ = μ)
        u
    end

    function compute_alpha(task)
        training = get_recommendee_split("explicit")
        params = read_params("$task/$source_name")
        u = train_model(training, params["λ"], mean(params["u"]), params["a"])
        preds = make_prediction(fill(1, num_items()), collect(1:num_items()), u, params["a"])
        write_recommendee_alpha(preds, "$task/$source_name")
    end
end
    
username = ARGS[1]
for task in ALL_TASKS
    compute_alpha(task)
end