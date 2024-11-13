import CodecZstd
import HTTP
import MsgPack
import NBInclude
import Oxygen
NBInclude.@nbinclude("notebooks/TrainingAlphas/Alpha.ipynb")
include("notebooks/TrainingAlphas/Baseline/get_user_biases.jl")

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

const PARAMS = Dict(
    m => read_params("baseline/v1/streaming/$m/rating") 
    for m in ["manga", "anime"]
)

Oxygen.@get "/heartbeat" function heartbeat(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

Oxygen.@get "/terminate" function terminate(req::HTTP.Request)
    Oxygen.terminate()
end

Oxygen.@post "/query" function query(req::HTTP.Request)
    dfs = Dict(k => RatingsDataset(v) for (k, v) in unpack(req.body))
    # baseline
    ret = Dict()
    for m in ALL_MEDIUMS
        params = PARAMS[m]
        df = as_metric(dfs[m], "rating")
        user_bias = get_user_biases(df, params)
        userid = 1
        ret["baseline/v1/streaming/$m/rating"] = params["a"] .+ get(user_bias, userid, 0)
    end
    # bagofwords
    X = zeros(Float32, sum(num_items.(ALL_MEDIUMS)) * 2)
    idx = 0
    for metric in ["rating", "watch"]
        for m in ALL_MEDIUMS
            β = PARAMS[m]["β"]
            df = as_metric(dfs[m], metric)
            if metric == "rating"
                for i in 1:length(df.itemid)
                    df.metric[i] -= ret["baseline/v1/streaming/$m/rating"][df.itemid[i]] * β
                end
            end
            for (a, v) in zip(df.itemid, df.metric)
                X[idx + a] = v
            end
            idx += num_items(m)
        end
    end
    ret["inputs"] = X
    msgpack(ret)
end

function test_heartbeat(port)
    running = false
    while !running
        try
            ret = HTTP.get("http://localhost:$port/heartbeat")
            running = !HTTP.Messages.iserror(ret)
        catch e
            sleep(1)
        end
    end
end

function test_terminate(port)
    HTTP.get("http://localhost:$port/terminate")
end

function test_query_inputs()
    Dict(
        "manga" => Dict(
            "source" => Int32[],
            "medium" => Int32[],
            "userid" => Int32[],
            "itemid" => Int32[],
            "status" => Int32[],
            "rating" => Float32[],
            "updated_at" => Float64[],
            "created_at" => Float64[],
            "started_at" => Float64[],
            "finished_at" => Float64[],
            "update_order" => Int32[],
            "progress" => Float32[],
            "progress_volumes" => Float32[],
            "repeat_count" => Int32[],
            "priority" => Int32[],
            "sentiment" => Int32[],
         ),
        "anime" => Dict(
            "source" => Int32[],
            "medium" => Int32[],
            "userid" => Int32[],
            "itemid" => Int32[],
            "status" => Int32[],
            "rating" => Float32[],
            "updated_at" => Float64[],
            "created_at" => Float64[],
            "started_at" => Float64[],
            "finished_at" => Float64[],
            "update_order" => Int32[],
            "progress" => Float32[],
            "progress_volumes" => Float32[],
            "repeat_count" => Int32[],
            "priority" => Int32[],
            "sentiment" => Int32[],
         ),        
    )
end

function test_query(port)
    pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
    url = "http://localhost:$port/query"
    d = pack(test_query_inputs())
    headers = Dict(
        "Content-Type" => "application/msgpack",
        "Content-Length" => string(sizeof(d)),
    )
    ret = HTTP.post(url, headers, d)
    @assert !HTTP.Messages.iserror(ret)
end

if isempty(ARGS)
    port = 8080
else
    port = parse(Int, ARGS[1])        
end
Threads.@spawn begin
    test_heartbeat(port)
    test_query(port)
    if isempty(ARGS)
        test_terminate(port)
    end
end
Oxygen.serveparallel(; host="0.0.0.0", port=port, access_log=nothing)