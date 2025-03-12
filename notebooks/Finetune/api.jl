module embed

import Base64
import CSV
import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/stdout.jl")

const PORT = parse(Int, ARGS[1])
const bluegreen = read("../../data/finetune/bluegreen", String)
const URL = read("../../secrets/url.compute.$bluegreen.txt", String)

const allowed_origins = [ "Access-Control-Allow-Origin" => "*" ]
const cors_headers = [
    allowed_origins...,
    "Access-Control-Allow-Headers" => "*",
    "Access-Control-Allow-Methods" => "GET, POST"
]
function CorsHandler(handle)
    return function (req::HTTP.Request)
        if HTTP.method(req) == "OPTIONS"
            return HTTP.Response(200, cors_headers)
        else
            r = handle(req)
            append!(r.headers, allowed_origins)
            return r
        end
    end
end
const MIDDLEWARE = [CorsHandler]

Oxygen.@post "/update" function update_state(r::HTTP.Request)::HTTP.Response
    # TODO enable ZSTD or GZIP compression
    d = decode(r)
    if isempty(d["state"])
        state = Dict("medium" => 1)
    else
        state = MsgPack.unpack(Base64.base64decode(d["state"]))
    end
    action = d["action"]
    r_update = HTTP.post(
        "$URL/update",
        encode(Dict("state" => state, "action" => action), :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r_update)
        return HTTP.Response(r_update.status)
    end
    data = decode(r_update)
    data["state"] = Base64.base64encode(MsgPack.pack(data["state"]))
    HTTP.Response(200, encode(data, :json)...)
end

function compile(port::Integer)
    profiles = CSV.read("../../secrets/test.users.csv", DataFrames.DataFrame, stringtype = String)
    while true
        try
            r = HTTP.get("$URL/ready")
            break
        catch
            logtag("STARTUP", "waiting for $URL to startup")
            sleep(1)
        end
    end
    for (source, username) in zip(profiles.source, profiles.username)
        action = Dict("type" => "add_user", "source" => source, "username" => username)
        r = HTTP.post(
            "http://localhost:$PORT/update",
            encode(Dict("state" => "", "action" => action), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r)
            logerror("error $(r.status)")
            continue
        end
    end
end

include("../julia_utils/start_oxygen.jl")

end
