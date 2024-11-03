function get_user_biases(df, params)
    μ_a, λ_u, λ_a, λ_wu, λ_wa, λ_wt = params["λ"]
    λ_u, λ_a = exp.((λ_u, λ_a))
    λ_wt = sigmoid(λ_wt)
    a = params["a"]
    item_counts = params["a_counts"]

    function get_user_partition(users, threadid, num_threads)
        [i for i in 1:length(users) if (users[i] % num_threads) + 1 == threadid]
    end
    userids = Set(df.userid)
    user_bias = Dict{Int32,Float32}(x => 0 for x in userids)
    denom = Dict{Int32,Float32}(x => λ_u for x in userids)
    T = Threads.nthreads()
    @sync for t = 1:T
        Threads.@spawn begin
            @inbounds for i in get_user_partition(df.userid, t, T)
                w = (item_counts[df.itemid[i]]^λ_wa) * (λ_wt^(1 - df.updated_at[i]))
                user_bias[df.userid[i]] += (df.rating[i] - a[df.itemid[i]]) * w
                denom[df.userid[i]] += w
            end
        end
    end
    for k in userids
        @inbounds user_bias[k] /= denom[k]
    end
    user_bias
end;