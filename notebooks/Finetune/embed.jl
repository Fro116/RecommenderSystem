mutable struct RemoteModel
    pending::Channel
end

const models = Dict(
    (m, t) => RemoteModel(Channel(Inf)) for m in [0, 1] for
    t in ["retrieval", "ranking"]
)

function call_model_url(users, medium::Int, task::String)
    max_retries = 3
    for retry in 1:max_retries
        try
            r_embed = HTTP.post(
                "$MODEL_URL/embed?medium=$medium&task=$task",
                encode(Dict("users" => users), :msgpack, :gzip)...,
                connect_timeout = 1,
            )
            return decode(r_embed)["embeds"]
        catch e
            logerror("model $MODEL_URL/embed?medium=$medium&task=$task failed with error $e")
            if retry < max_retries
                sleep(1)
            end
        end
    end
    nothing
end

function run_model(medium::Int, task::String)
    batch_size = 16
    model = models[(medium, task)]
    while true
        batch = []
        first_req = take!(model.pending)
        push!(batch, first_req)
        while length(batch) < batch_size && isready(model.pending)
            req = take!(model.pending)
            push!(batch, req)
        end
        users = [u for (u, _, _, _) in batch]
        d_embed = call_model_url(users, medium, task)
        for i = 1:length(batch)
            _, _, _, result_channel = batch[i]
            if isnothing(d_embed)
                put!(result_channel, nothing)
            else
                put!(result_channel, d_embed[i])
            end
        end
    end
end

function query_model(
    user,
    medium::Int,
    test_matchedids::Union{Vector,Nothing};
)
    if isnothing(test_matchedids)
        task = "retrieval"
    else
        task = "ranking"
        user = copy(user)
        user["ranking_items"] = test_matchedids
    end
    model = models[(medium, task)]
    result_channel = Channel(1)
    put!(model.pending, (user, medium, task, result_channel))
    ret = take!(result_channel)
    if isnothing(ret)
        return nothing
    end
    data = Dict()
    for (k, v) in ret
        data[k] = Float32.(v)
    end
    data
end

for m in [0, 1]
    for task in ["retrieval", "ranking"]
        Threads.@spawn run_model(m, task)
    end
end

function compute_retrieval(registry, medium::Int, user, idxs)
    masked_logits = registry["$medium.watch.weight"] * user["$medium.retrieval"]
    p_masked = softmax(masked_logits)[idxs]
    p = sum(registry["$medium.retrieval.coefs"] .* [p_masked])
end

function compute_ranking(registry, medium::Int, user)
    r_masked = user["$medium.ranking"]
    r_baseline = fill(registry["$medium.rating_mean"], length(r_masked))
    r = sum(registry["$medium.rating.coefs"] .* [r_baseline, r_masked])
end
