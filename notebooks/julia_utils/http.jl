import HTTP
import JSON3
import MsgPack
import CodecZlib
import CodecZstd

function encode(d::Dict, content_type::Symbol, encoding::Union{Symbol, Nothing} = nothing)
    if content_type == :json
        headers = Dict("Content-Type" => "application/json")
        body = Vector{UInt8}(JSON3.write(d))
    elseif content_type == :msgpack
        headers = Dict("Content-Type" => "application/msgpack")
        body = Vector{UInt8}(MsgPack.pack(d))
    else
        @assert false
    end
    if encoding == :zstd
        headers["Content-Encoding"] = "zstd"
        body = CodecZstd.transcode(CodecZstd.ZstdCompressor, body)
    elseif encoding == :gzip
        headers["Content-Encoding"] = "gzip"
        body = CodecZstd.transcode(CodecZlib.GzipCompressor, body)
    else
        @assert isnothing(encoding)
    end
    headers, body
end

function decode(r::HTTP.Message)::Dict
    body = r.body
    if HTTP.headercontains(r, "Content-Encoding", "zstd")
        body = CodecZstd.transcode(CodecZstd.ZstdDecompressor, body)
    elseif HTTP.headercontains(r, "Content-Encoding", "gzip")
        body = CodecZlib.transcode(CodecZlib.GzipDecompressor, body)
    else
        @assert !HTTP.hasheader(r, "Content-Encoding")
    end
    if HTTP.headercontains(r, "Content-Type", "application/json")
        return JSON3.read(String(body), Dict{String,Any})
    elseif HTTP.headercontains(r, "Content-Type", "application/msgpack")
        return MsgPack.unpack(body)
    elseif HTTP.headercontains(r, "Content-Type", "application/octet-stream")
        return body
    else
        @assert false
    end
end

function get_preferred_encoding(r::HTTP.Request)
    if occursin("zstd", HTTP.header(r, "Accept-Encoding", ""))
        return :zstd
    elseif occursin("gzip", HTTP.header(r, "Accept-Encoding", ""))
        return :gzip
    else
        return nothing
    end
end
