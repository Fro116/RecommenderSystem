module embed

import Oxygen
include("../Training/import_list.jl")
include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

const PORT = parse(Int, ARGS[1])
const FETCH_URL = ARGS[2]
const MODEL_URL = ARGS[3]
const datadir = "../../data/finetune"

standardize(x::Dict) = Dict(lowercase(String(k)) => v for (k, v) in x)

Oxygen.@post "/embed_user" function embed_user(r::HTTP.Request)::HTTP.Response
    d = decode(r)
    r_read = HTTP.post(
        "$FETCH_URL/read",
        encode(Dict("source" => d["source"], "username" => d["username"]), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_read)
        return HTTP.Response(r_read.status, [])
    end
    d_read = decode(r_read)
    data = decompress(d_read["data"])
    data["user"] = standardize(data["user"])
    data["items"] = standardize.(data["items"])
    u = import_user(d_read["source"], data)
    r_embed = HTTP.post(
        "$MODEL_URL/embed",
        encode(u, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_embed)
        return HTTP.Response(r_embed.status, [])
    end    
    HTTP.Response(200, encode(decode(r_embed), :msgpack)...)
end

function compile(port::Integer)
    profiles = CSV.read("../../secrets/test.users.csv", DataFrames.DataFrame, stringtype = String)
    while true
        try
            r = HTTP.get("$MODEL_URL/ready")
            break
        catch
            logtag("STARTUP", "waiting for $MODEL_URL to startup")
            sleep(1)
        end
    end
    @sync for (source, username) in zip(profiles.source, profiles.username)
        Threads.@spawn HTTP.post(
            "http://localhost:$PORT/embed_user",
            encode(Dict("source" => source, "username" => username), :msgpack)...,
        )
    end
end

include("../julia_utils/start_oxygen.jl")

end
