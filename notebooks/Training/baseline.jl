import CSV
import DataFrames
import Glob
import MsgPack
import Memoize: @memoize
import NNlib: sigmoid
import Optim
import ProgressMeter: @showprogress
import Random
import StatsBase

include("../julia_utils/stdout.jl")

const datadir = "../../data/training"
const MEDIUM_MAP = Dict(0 => "manga", 1 => "anime")
const metric = ARGS[1]

@memoize function num_items(medium::Int)
    m = MEDIUM_MAP[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

loss(x, y, w) = sum((x - y) .^ 2 .* w) / sum(w)

@kwdef struct RatingsDataset
    userid::Vector{Int32} = []
    itemid::Vector{Int32} = []
    rating::Vector{Float32} = []
end

function subset(x::RatingsDataset, ord)
    mask(arr) = !isempty(arr) ? arr[ord] : arr
    RatingsDataset([mask(getfield(x, c)) for c in fieldnames(RatingsDataset)]...)
end

function get_data(split::String, medium::Int)
    userids = []
    itemids = []
    ratings = []
    userid_base = 0
    @showprogress for outdir in readdir("$datadir/users/$split")
        fns = Glob.glob("$datadir/users/$split/$outdir/*.msgpack")
        push!(userids, Vector{Vector{Int32}}(undef, length(fns)))
        push!(itemids, Vector{Vector{Int32}}(undef, length(fns)))
        push!(ratings, Vector{Vector{Float32}}(undef, length(fns)))
        Threads.@threads for (i, f) in collect(Iterators.enumerate(fns))
            user = MsgPack.unpack(read(f))
            items = [x for x in user["items"] if x["medium"] == medium && x[metric] != 0]
            userids[end][i] = fill(userid_base + i, length(items))
            itemids[end][i] = [x["matchedid"] for x in items]
            ratings[end][i] = [x[metric] for x in items]
        end
        userid_base += length(fns)
    end
    N = sum(sum(length.(x)) for x in userids)
    rd = RatingsDataset(
        Vector{Int32}(undef, N),
        Vector{Int32}(undef, N),
        Vector{Float32}(undef, N),
    )
    idx = 0
    @showprogress for p = 1:length(userids)
        for i = 1:length(userids[p])
            Threads.@threads for j = 1:length(userids[p][i])
                rd.userid[idx+j] = userids[p][i][j]
                rd.itemid[idx+j] = itemids[p][i][j]
                rd.rating[idx+j] = ratings[p][i][j]
            end
            idx += length(userids[p][i])
        end
    end
    rd.itemid .+= 1
    rd
end

function training_test_split(df::RatingsDataset, test_frac::Float64)
    userids = Random.shuffle(sort(collect(Set(df.userid))))
    n_train = Int(round(length(userids) * (1 - test_frac)))
    train_userids = Set(userids[1:n_train])
    test_userids = Set(userids[n_train+1:end])
    train_df = subset(df, df.userid .∈ (train_userids,))
    test_df = subset(df, df.userid .∈ (test_userids,))
    train_df, test_df
end

function random_split(df::RatingsDataset, test_frac::Float64)
    mask = rand(length(df.userid)) .< test_frac
    train_df = subset(df, .!mask)
    test_df = subset(df, mask)
    train_df, test_df
end

function get_user_biases(df, λ, a, user_countmap, item_countmap)
    μ_a, λ_u, λ_a, λ_wu, λ_wa = λ
    λ_u, λ_a = exp.((λ_u, λ_a))
    function get_user_partition(users, threadid, num_threads)
        [i for i = 1:length(users) if (users[i] % num_threads) + 1 == threadid]
    end
    userids = Set(df.userid)
    user_bias = Dict{Int32,eltype(a)}(x => zero(eltype(a)) for x in userids)
    denom = Dict{Int32,typeof(λ_u)}(x => λ_u for x in userids)
    T = Threads.nthreads()
    @sync for t = 1:T
        Threads.@spawn begin
            @inbounds for i in get_user_partition(df.userid, t, T)
                w =
                    (user_countmap[df.userid[i]]^λ_wu) *
                    (get(item_countmap, df.itemid[i], 1)^λ_wa)
                user_bias[df.userid[i]] += (df.rating[i] - a[df.itemid[i]]) * w
                denom[df.userid[i]] += w
            end
        end
    end
    for k in userids
        @inbounds user_bias[k] /= denom[k]
    end
    user_bias
end

@memoize function get_user_partition(users, threadid, num_threads)
    [i for i = 1:length(users) if (users[i] % num_threads) + 1 == threadid]
end

function update_users!(users, items, ratings, weights, u, a, μ_uλ_u, Ω)
    Threads.@threads for i = 1:length(u)
        @inbounds u[i] = μ_uλ_u
    end
    T = Threads.nthreads()
    @sync for t = 1:T
        Threads.@spawn begin
            @inbounds for row in get_user_partition(users, t, T)
                i = users[row]
                j = items[row]
                r = ratings[row]
                w = weights[row]
                u[i] += (r - a[j]) * w
            end
        end
    end
    Threads.@threads for i = 1:length(u)
        @inbounds u[i] /= Ω[i]
    end
end

@memoize function get_countmap(df, col)
    StatsBase.countmap(getfield(df, col))
end

@memoize function get_counts(df, col)
    data = getfield(df, col)
    counts = StatsBase.countmap(data)
    [counts[x] for x in data]
end

function get_weights(df, λ_wu, λ_wa)
    users = get_counts(df, :userid)
    items = get_counts(df, :itemid)
    w = Vector{typeof(λ_wu)}(undef, length(users))
    Threads.@threads for i = 1:length(w)
        w[i] = (users[i]^λ_wu) * (items[i]^λ_wa)
    end
    w
end;

function get_denom(weights, λ, users, num_users)
    Ω_u = Vector{eltype(weights)}(undef, num_users)
    Threads.@threads for i = 1:length(Ω_u)
        Ω_u[i] = λ
    end
    T = Threads.nthreads()
    @sync for t = 1:T
        Threads.@spawn begin
            @inbounds for row in get_user_partition(users, t, T)
                Ω_u[users[row]] += weights[row]
            end
        end
    end
    Ω_u
end

function train_model(λ, training, medium)
    μ_a, λ_u, λ_a, λ_wu, λ_wa = λ
    λ_u, λ_a = exp.((λ_u, λ_a))
    users, items, ratings = training.userid, training.itemid, training.rating
    weights = get_weights(training, λ_wu, λ_wa)
    u = zeros(typeof(λ_u), maximum(users))
    a = zeros(typeof(λ_a), num_items(medium))
    Ω_u = get_denom(weights, λ_u, users, length(u))
    Ω_a = get_denom(weights, λ_a, items, length(a))
    max_iters = 8
    @showprogress for _ = 1:max_iters
        update_users!(items, users, ratings, weights, a, u, μ_a * λ_a, Ω_a)
        update_users!(users, items, ratings, weights, u, a, 0, Ω_u)
    end
    u, a
end;

function make_prediction(users, items, u, a)
    r = Array{eltype(u)}(undef, length(users))
    Threads.@threads for i = 1:length(r)
        @inbounds r[i] = u[users[i]] + a[items[i]]
    end
    r
end;

function mse_and_beta(λ, training, test_input, test_output, medium)
    _, a = train_model(λ, training, medium)
    u = get_user_biases(
        test_input,
        λ,
        a,
        get_countmap(test_input, :userid),
        get_countmap(training, :itemid),
    )
    x = Array{eltype(a)}(undef, length(test_output.userid))
    Threads.@threads for i = 1:length(x)
        @inbounds x[i] = get(u, test_output.userid[i], 0) + a[test_output.itemid[i]]
    end
    y = test_output.rating
    w = [1 / c for c in get_counts(test_output, :userid)]
    xw = (x .* sqrt.(w))
    yw = (y .* sqrt.(w))
    β = (xw'xw + 1.0f-9) \ xw'yw
    L = loss(x * β, y, w)
    L, β
end;

function average_item_rating(df)
    s = Dict()
    w = Dict()
    for (a, r) in zip(df.itemid, df.rating)
        if a ∉ keys(w)
            s[a] = 0
            w[a] = 0
        end
        s[a] += r
        w[a] += 1
    end
    StatsBase.mean([s[a] / w[a] for a in keys(w)])
end

function save_model(medium::Int)
    training = get_data("training", medium)
    training, test = training_test_split(training, 0.1)
    test_input, test_output = random_split(test, 0.1)
    test = nothing
    λ = Float32[average_item_rating(training), 0, 0, -1, 0]
    res = Optim.optimize(
        λ -> mse_and_beta(λ, training, test_input, test_output, medium)[1],
        Float32[average_item_rating(training), 0, 0, -1, 0],
        Optim.LBFGS(),
        autodiff = :forward,
        Optim.Options(
            show_trace = true,
            extended_trace = true,
            g_tol = Float64(sqrt(eps(Float32))),
            time_limit = 1800,
        ),
    )
    λ = Optim.minimizer(res)
    mse, β = mse_and_beta(λ, training, test_input, test_output, medium)
    logtag("BASELINE", "The optimal λ, mse is $λ, $mse")
    _, a = train_model(λ, training, medium)
    d = Dict(
        "params" => Dict(
            "λ" => λ,
            "a" => a,
            "item_counts" => get_countmap(training, :itemid),
        ),
        "weight" => [β],
        "bias" => a .* β,
    )
    outfn = "$datadir/baseline.$(metric).$(medium).msgpack"
    open(outfn, "w") do f
        write(f, MsgPack.pack(d))
    end
    template = raw"tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => outfn,
        "{OUTPUT}" => "baseline.$(metric).$(medium).msgpack",
    )
    run(`sh -c $cmd`)
end

for m in keys(MEDIUM_MAP)
    save_model(m)
end
