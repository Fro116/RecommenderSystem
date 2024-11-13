import CodecZstd
import HTTP
import MsgPack
import NBInclude
import Oxygen
import SparseArrays
NBInclude.@nbinclude("notebooks/TrainingAlphas/Alpha.ipynb")

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
    (m, x) => read_params("nondirectional/v1/training/$m/watch/$x")["S"] for
    m in ["manga", "anime"] for
    x in ["adaptations", "dependencies", "recaps", "related"]
)

Oxygen.@get "/heartbeat" function heartbeat(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

Oxygen.@get "/terminate" function terminate(req::HTTP.Request)
    Oxygen.terminate()
end

Oxygen.@post "/query" function query(req::HTTP.Request)
    dfs = Dict(k => RatingsDataset(v) for (k, v) in unpack(req.body))
    spvec(df, m) =
        SparseArrays.sparsevec(df.itemid, fill(1, length(df.itemid)), num_items(m))
    finished = Dict(
        k => spvec(subset(v, v.status .>= get_status(:completed)), k) for (k, v) in dfs
    )
    started = Dict(
        k => spvec(
            subset(
                v,
                (v.status .>= get_status(:dropped)) .&& (v.status .!= get_status(:planned)),
            ),
            k,
        ) for (k, v) in dfs
    )
    ret = Dict()
    mediat = Dict("anime" => "manga", "manga" => "anime")
    for m in ALL_MEDIUMS
        adaptation_blacklist = (PARAMS[(m, "adaptations")] * started[mediat[m]]) .> 0
        adaptation_whitelist = (PARAMS[(m, "related")] * started[m]) .> 0
        recap_blacklist = (PARAMS[(m, "recaps")] * started[m]) .> 0
        recap_whitelist = (PARAMS[(m, "dependencies")] * finished[m]) .> 0
        dependency_blacklist =
            (PARAMS[(m, "dependencies")] * (ones(Float32, num_items(m)) - finished[m])) .> 0
        dependency_whitelist = (PARAMS[(m, "dependencies")] * finished[m]) .> 0
        ret["nondirectional/$m/blacklist"] = collect(
            @. (
                (adaptation_blacklist && !adaptation_whitelist) ||
                (recap_blacklist && !recap_whitelist) ||
                (dependency_blacklist && !dependency_whitelist)
            )
        )
        ret["nondirectional/$m/seen"] = collect(started[m] .> 0)
    end
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