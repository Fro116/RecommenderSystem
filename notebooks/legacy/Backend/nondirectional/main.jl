import CodecZstd
import HTTP
import MsgPack
import NBInclude
import Oxygen
import SparseArrays
NBInclude.@nbinclude("notebooks/Train/Alpha.ipynb")

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    headers = Dict("Content-Type" => "application/msgpack")
    HTTP.Response(200, headers, body)
end

const PARAMS = Dict(
    (m, x) => read_params("nondirectional/v1/training/$m/watch/$x")["S"] for
    m in ALL_MEDIUMS for
    x in ["adaptations", "dependencies", "recaps", "related"]
)

function get_mappings(medium)
    df = read_csv(get_data_path("processed_data/$medium.mapping.csv"))
    Dict(k => df[:, k] for k in DataFrames.names(df))
end

const MAPPINGS = Dict(m => get_mappings(m) for m in ALL_MEDIUMS)

READY::Bool = false

Oxygen.@get "/ready" function ready(req::HTTP.Request)
    if !READY
        return HTTP.Response(503, [])
    else
        return HTTP.Response(200, [])
    end
end

Oxygen.@get "/terminate" terminate(req::HTTP.Request) = Oxygen.terminate()

Oxygen.@post "/query" function query(req::HTTP.Request)
    if !READY && !HTTP.Messages.hasheader(req, "Startup")
        return HTTP.Response(503, [])
    end
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
    planned = Dict(
        k => spvec(subset(v, v.status .== get_status(:planned)), k) for (k, v) in dfs
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
        ret["nondirectional/$m/started"] = collect(started[m] .> 0)
        ret["nondirectional/$m/planned"] = collect(planned[m] .> 0)
        ret["nondirectional/$m/mapping"] = MAPPINGS[m]
    end
    msgpack(ret)
end

function test_ready(port)
    running = false
    while !running
        try
            ret = HTTP.get("http://localhost:$port/ready"; status_exception = false)
            running = ret.status == 503
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
    body = pack(test_query_inputs())
    headers = Dict(
        "Content-Type" => "application/msgpack",
        "Startup" => true,
    )
    ret = HTTP.post(url, headers, body)
end

if isempty(ARGS)
    port = 8080
else
    port = parse(Int, ARGS[1])        
end
Threads.@spawn begin
    test_ready(port)
    test_query(port)
    if isempty(ARGS)
        test_terminate(port)
    end
    global READY
    READY = true
end
Oxygen.serveparallel(; host="0.0.0.0", port=port, access_log=nothing)
