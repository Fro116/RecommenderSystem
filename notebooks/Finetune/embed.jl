mutable struct RemoteModel
    pending::Channel
end

const models = Dict(
    (m, t) => RemoteModel(Channel(Inf)) for m in [0, 1] for
    t in ["retrieval", "ranking"]
)

function run_model(medium::Integer, task::AbstractString)
    batch_size = 16
    model = models[(medium, task)]
    while true
        batch = []
        first_req = take!(model.pending)
        if first_req[2] != medium || first_req[3] != task
            put!(model.pending, first_req)
            yield()
            continue
        end
        push!(batch, first_req)
        while length(batch) < batch_size && isready(model.pending)
            req = take!(model.pending)
            if req[2] == medium && req[3] == task
                push!(batch, req)
            else
                put!(model.pending, req)
                break
            end
        end
        try
            users = [u for (u, _, _, _) in batch]
            r_embed = HTTP.post(
                "$MODEL_URL/embed?medium=$medium&task=$task",
                encode(Dict("users" => users), :msgpack)...,
                connect_timeout = 1,
            )
            d_embed = decode(r_embed)["embeds"]
            for i = 1:length(batch)
                _, _, _, result_channel = batch[i]
                put!(result_channel, (d_embed[i], time()))
            end
        catch e
            logerror("model $medium $task failed with error $e")
            for i = 1:length(batch)
                _, _, _, result_channel = batch[i]
                put!(result_channel, (nothing, time()))
            end
        end
    end
end

function query_model(
    user,
    medium::Integer,
    test_matchedids::Union{Vector,Nothing};
    timeout::Real = 10,
    retries::Int = 1,
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
    retry() =
        retries == 0 ? nothing :
        query_model(user, medium, test_matchedids; timeout = timeout, retries = retries - 1)
    model = models[(medium, task)]
    result_channel = Channel(1)
    put!(model.pending, (user, medium, task, result_channel))
    timed_out = false
    ret = nothing
    timer = Timer(timeout)
    try
        ret = fetch(Threads.@spawn take!(result_channel))
    catch e
        timed_out = true
    finally
        close(timer)
    end
    if timed_out
        return retry()
    end
    if !isnothing(ret) && !isnothing(ret[1])
        data = Dict()
        for (k, v) in ret[1]
            data[k] = k == "version" ? v : Float32.(v)
        end
        return data
    end
    retry()
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