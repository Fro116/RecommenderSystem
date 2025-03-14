import CodecZstd
import HTTP
import MsgPack
import NBInclude
import Oxygen
NBInclude.@nbinclude("notebooks/Train/Alpha.ipynb")

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    headers = Dict("Content-Type" => "application/msgpack")
    HTTP.Response(200, headers, body)
end

READY::Bool = false

Oxygen.@get "/ready" function ready(req::HTTP.Request)
    if !READY
        return HTTP.Response(503, [])
    else
        return HTTP.Response(200, [])
    end
end

Oxygen.@get "/terminate" terminate(req::HTTP.Request) = Oxygen.terminate()

function outputs()
    d = Dict()
    for medium in ALL_MEDIUMS
        d["baseline/v1/streaming/$medium/rating"] = zeros(Float32, num_items(medium))
        for metric in ALL_METRICS
            d["bagofwords/v1/streaming/$medium/$metric"] = zeros(Float32, num_items(medium))
        end
    end
    d["inputs"] = zeros(Float32, sum(num_items.(ALL_MEDIUMS)) * 2)
    msgpack(d)
end

Oxygen.@get "/query" query(req::HTTP.Request) = outputs();
Oxygen.@post "/query" query(req::HTTP.Request) = outputs();
Oxygen.serveparallel(; host="0.0.0.0", port=parse(Int, ARGS[1]), access_log=nothing)
