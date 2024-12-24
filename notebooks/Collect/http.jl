import HTTP
import JSON3

function encode(d::Dict, encoding::Symbol)
    if encoding == :json
        headers = Dict("Content-Type" => "application/json")
        body = Vector{UInt8}(JSON3.write(d))
    else
        @assert false
    end
    headers, body
end

function decode(r::HTTP.Message)::Dict
    if HTTP.headercontains(r, "Content-Type", "application/json")
        return JSON3.read(String(r.body), Dict{String,Any})
    else
        @assert false
    end
end