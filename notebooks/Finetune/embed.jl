mutable struct RemoteModel
    lock::ReentrantLock
    pending::Vector
    finished::Dict
    reqid::Int
end

const models = Dict(
    (m, t) => RemoteModel(ReentrantLock(), [], Dict(), 0) for m in [0, 1] for
    t in ["retrieval", "ranking"]
)

function run_model(medium::Integer, task::AbstractString)
    batch_size = 16
    batch = []
    model = models[(medium, task)]
    while true
        lock(model.lock) do
            if isempty(model.pending)
                return
            end
            for i = 1:length(model.pending)
                reqid, user, m, t = model.pending[i]
                if m == medium && t == task && length(batch) < batch_size
                    push!(batch, model.pending[i])
                    model.pending[i] = nothing
                end
            end
            filter!(x -> !isnothing(x), model.pending)
        end
        if isempty(batch)
            sleep(50 * 0.001)
            continue
        end
        try
            users = [u for (_, u, _, _) in batch]
            r_embed = HTTP.post(
                "$MODEL_URL/embed?medium=$medium&task=$task",
                encode(Dict("users" => users), :msgpack)...,
                connect_timeout=1,
            )
            d_embed = decode(r_embed)["embeds"]
            lock(model.lock) do
                for i = 1:length(batch)
                    model.finished[batch[i][1]] = (d_embed[i], time())
                end
            end
        catch e
            logerror("model $medium $task failed with error $e")
            lock(model.lock) do
                for i = 1:length(batch)
                    model.finished[batch[i][1]] = (nothing, time())
                end
            end
        end
        empty!(batch)
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
    reqid = lock(model.lock) do
        model.reqid += 1
        reqid = model.reqid
        push!(model.pending, (reqid, user, medium, task))
        reqid
    end
    t = time()
    while time() - t < timeout
        sleep(50 * 0.001)
        ret = lock(model.lock) do
            if reqid in keys(model.finished)
                return pop!(model.finished, reqid)
            else
                return nothing
            end
        end
        if !isnothing(ret)
            data = Dict()
            for (k, v) in ret[1]
                if k == "version"
                    data[k] = v
                else
                    data[k] = Float32.(v)
                end
            end
            return data
        end
    end
    lock(model.lock) do
        ts = time()
        for k in keys(model.finished)
            v, vtime = model.finished[k]
            if isnothing(v) && ts - vtime > 300
                pop!(model.finished, k)
            end
        end
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
