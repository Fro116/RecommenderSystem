mutable struct RemoteModel
    pending::Channel
end

const models = Dict(
    (m, t) => RemoteModel(Channel(Inf)) for m in [0, 1] for
    t in ["retrieval", "ranking"]
)

function call_model_url(users, medium, task)
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

function run_model(medium::Integer, task::AbstractString)
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
    medium::Integer,
    test_matchedids::Union{Vector,Nothing};
)
    if isnothing(test_matchedids)
        task = "retrieval"
    else
        task = "ranking"
        ts = user["timestamp"]
        test_items = []
        ranking_items = get_ranking_items(medium, user["user"]["source"])
        for idx in test_matchedids
            item = copy(ranking_items[idx])
            item["history_max_ts"] = ts
            push!(test_items, item)
        end
        user = copy(user)
        user["test_items"] = test_items
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
        data[k] = k == "version" ? v : Float32.(v)
    end
    data
end

@memoize function num_items(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

@memoize function get_ranking_items(medium, source)
    info = Dict()
    for i = 1:num_items(medium)
        info[i-1] = Dict{String,Any}(
            "medium" => medium,
            "matchedid" => -1,
            "distinctid" => -1,
            "status" => 0,
            "rating" => 0,
            "progress" => 0,
        )
    end
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks = 1)
    for match_source in [true, false]
        for i = 1:DataFrames.nrow(df)
            if (df.source[i] != source && match_source) ||
               info[df.matchedid[i]]["matchedid"] != -1
                continue
            end
            info[df.matchedid[i]] = Dict{String,Any}(
                "medium" => medium,
                "matchedid" => df.matchedid[i],
                "distinctid" => df.distinctid[i],
                "status" => 0,
                "rating" => 0,
                "progress" => 0,
            )
        end
    end
    info
end

for m in [0, 1]
    for task in ["retrieval", "ranking"]
        Threads.@spawn run_model(m, task)
    end
end