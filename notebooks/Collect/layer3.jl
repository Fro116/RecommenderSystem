const PORT = parse(Int, ARGS[1])
const LAYER_2_URL = ARGS[2]
const TOKEN_TIMEOUT = parse(Int, ARGS[3])

import Dates
import HTTP
import JSON3
import Oxygen
import UUIDs

include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")

struct Response
    status::Int
    body::Vector{UInt8}
    headers::Dict{String,String}
end

function decode(r::Response)::Dict
    if r.headers["Content-Type"] == "application/json"
        return JSON3.read(String(r.body), Dict{String,Any})
    else
        @assert false
    end
end

function request(query::String, params::Dict{String,Any})
    for delay in ExponentialBackOff(;
        n = 10,
        first_delay = 1,
        max_delay = 1000,
        factor = 2.0,
        jitter = 0.1,
    )
        r = HTTP.post(
            "$LAYER_2_URL/$query",
            encode(params, :json)...,
            status_exception = false,
        )
        if r.status < 400 || r.status in [400, 401, 403, 404]
            # 400, 401 -> auth error
            # 403 -> list is private
            # 404 -> invalid user
            return Response(r.status, r.body, Dict(k => v for (k, v) in r.headers))
        end
        if HTTP.hasheader(r.headers, "Retry-After")
            try
                retry_after = first(HTTP.headers(r, "Retry-After"))
                retry_timeout = min(parse(Int, retry_after), 1000)
                delay = max(delay, retry_timeout)
            catch
                logerror("invalid Retry-After $retry_after for $query $params")
            end
        end
        logerror("retrying $query $params after $delay seconds")
        sleep(delay)
    end
    logerror("could not retrieve $query $params")
    Response(500, [], Dict())
end

@enum Errors begin
    INVALID_SESSION = 401
    NOT_FOUND = 404
    TOKEN_UNAVAIABLE = 503
end

function retry(f::Function; count = 10)
    retryable = Errors[INVALID_SESSION, TOKEN_UNAVAIABLE]
    x = f()
    for _ = 1:count-1
        if x ∉ retryable
            return x
        end
        logerror("retrying function after error $x")
        sleep(1)
        x = f()
    end
    x
end

const SESSIONS = Dict{Any,Any}() # token -> (sessionid, access_time)
const SESSIONS_LOCK = ReentrantLock()

function get_session(token)
    auth = lock(SESSIONS_LOCK) do
        if token in keys(SESSIONS)
            sessionid, _ = SESSIONS[token]
            SESSIONS[token] = (sessionid, time())
            return sessionid
        end
        nothing
    end
    if !isnothing(auth)
        return auth
    end
    if token["resource"]["location"] == "kitsu"
        r = request("kitsu", Dict("token" => token, "endpoint" => "token"))
        if r.status >= 400
            return TOKEN_UNAVAIABLE::Errors
        end
        data = decode(r)
        sessionid = data["token"]
    elseif token["resource"]["location"] == "animeplanet"
        sessionid = string(UUIDs.uuid4())
    else
        @assert false
    end
    lock(SESSIONS_LOCK) do
        SESSIONS[token] = (sessionid, time())
        if length(SESSIONS) > 10_000
            ks = collect(keys(SESSIONS))
            for k in ks
                _, access_time = SESSIONS[k]
                if time() - access_time > 10_000
                    delete!(SESSIONS, k)
                end
            end
        end
        sessionid
    end
end

function invalidate_session(token::Dict)
    lock(SESSIONS_LOCK) do
        delete!(SESSIONS, token)
    end
end

Oxygen.@post "/mal_username" function mal_username(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    data = retry(() -> get_malweb_username(userid))
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :json)...)
end

function get_malweb_username(userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "malweb", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "malweb",
            Dict("token" => token, "endpoint" => "username", "userid" => userid),
        )
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/mal_user" function mal_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    user_data = retry(() -> get_malweb_user(username))
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)_count"] == 0
            continue
        end
        list = retry(() -> get_mal_list(m, username))
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    userid_data = Dict([x => user_data[x] for x in ["username", "userid", "version"]]...)
    ret = Dict("user" => user_data, "items" => items, "userid" => userid_data)
    HTTP.Response(200, encode(ret, :json)...)
end

function get_malweb_user(username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "malweb", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "malweb",
            Dict("token" => token, "endpoint" => "user", "username" => username),
        )
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_mal_list(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "mal", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        entries = []
        offset = 0
        while true
            s = request(
                "mal",
                Dict(
                    "token" => token,
                    "endpoint" => "list",
                    "username" => username,
                    "medium" => medium,
                    "offset" => offset,
                ),
            )
            if s.status >= 400
                return NOT_FOUND::Errors
            end
            data = decode(s)
            append!(entries, data["entries"])
            if "next" ∉ keys(data)
                return entries
            end
            offset += data["limit"]
        end
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/mal_media" function mal_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    details = retry(() -> get_mal_media("mal", medium, itemid))
    if isa(details, Errors)
        return HTTP.Response(Int(details), [])
    end
    relations = retry(() -> get_mal_media("malweb", medium, itemid))
    if isa(relations, Errors)
        return HTTP.Response(Int(relations), [])
    end
    HTTP.Response(200, encode(merge(details, relations), :json)...)
end

function get_mal_media(location::String, medium::String, itemid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => location, "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            location,
            Dict(
                "token" => token,
                "endpoint" => "media",
                "medium" => medium,
                "itemid" => itemid,
            ),
        )
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/anilist_user" function anilist_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    user_data = retry(() -> get_anilist_user(userid))
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)Count"] == 0
            continue
        end
        list = retry(() -> get_anilist_list(m, userid))
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    ret = Dict("user" => user_data, "items" => items)
    HTTP.Response(200, encode(ret, :json)...)
end

function get_anilist_user(userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "anilist",
            Dict("token" => token, "endpoint" => "user", "userid" => userid),
        )
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_anilist_list(medium::String, userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        entries = []
        chunk = 1
        while true
            s = request(
                "anilist",
                Dict(
                    "token" => token,
                    "endpoint" => "list",
                    "userid" => userid,
                    "medium" => medium,
                    "chunk" => chunk,
                ),
            )
            if s.status >= 400
                return NOT_FOUND::Errors
            end
            data = decode(s)
            append!(entries, data["entries"])
            if !data["next"]
                return entries
            end
            chunk += 1
        end
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/anilist_media" function anilist_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = retry(() -> get_anilist_media(medium, itemid))
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :json)...)
end

function get_anilist_media(medium::String, itemid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "anilist",
            Dict(
                "token" => token,
                "endpoint" => "media",
                "medium" => medium,
                "itemid" => itemid,
            ),
        )
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/kitsu_user" function kitsu_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    user_data = retry(() -> get_kitsu_user(userid))
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    ret = Dict{String,Any}("user" => user_data)
    seconds_in_day = 86400 # allow one day of fudge
    updated_at = try
        Dates.datetime2unix(
            Dates.DateTime(user_data["updatedAt"], "yyyy-mm-ddTHH:MM:SS.sssZ"),
        )
    catch
        logerror("""kitsu_user could not parse time $(user_data["updatedAt"])""")
        nothing
    end
    items = []
    if (user_data["manga_count"] + user_data["anime_count"]) > 0
        list = retry(() -> get_kitsu_list(userid))
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    ret = Dict("user" => user_data, "items" => items)
    HTTP.Response(200, encode(ret, :json)...)
end

function get_kitsu_user(userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        auth = get_session(token)
        if isa(auth, Errors)
            return auth
        end
        s = request(
            "kitsu",
            Dict(
                "token" => token,
                "auth" => auth,
                "endpoint" => "user",
                "userid" => userid,
            ),
        )
        if s.status in [400, 401]
            invalidate_session(token)
            return INVALID_SESSION::Errors
        elseif s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_kitsu_list(userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        auth = get_session(token)
        if isa(auth, Errors)
            return auth
        end
        entries = []
        offset = 0
        while true
            s = request(
                "kitsu",
                Dict(
                    "token" => token,
                    "endpoint" => "list",
                    "auth" => auth,
                    "userid" => userid,
                    "offset" => offset,
                ),
            )
            if s.status in [400, 401]
                invalidate_session(token)
                return INVALID_SESSION::Errors
            elseif s.status >= 400
                return NOT_FOUND::Errors
            end
            data = decode(s)
            append!(entries, data["entries"])
            if "next" ∉ keys(data)
                return entries
            end
            offset += data["limit"]
        end
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/kitsu_media" function kitsu_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = retry(() -> get_kitsu_media(medium, itemid))
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :json)...)
end

function get_kitsu_media(medium::String, itemid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        auth = get_session(token)
        if isa(auth, Errors)
            return auth
        end
        s = request(
            "kitsu",
            Dict(
                "token" => token,
                "auth" => auth,
                "endpoint" => "media",
                "medium" => medium,
                "itemid" => itemid,
            ),
        )
        if s.status in [400, 401]
            invalidate_session(token)
            return INVALID_SESSION::Errors
        elseif s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/animeplanet_username" function animeplanet_username(
    r::HTTP.Request,
)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    data = retry(() -> get_animeplanet_username(userid))
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :json)...)
end

function get_animeplanet_username(userid::Int)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        s = request(
            "animeplanet",
            Dict(
                "token" => token,
                "sessionid" => sessionid,
                "endpoint" => "username",
                "userid" => userid,
            ),
        )
        if s.status == 401
            invalidate_session(token)
            return INVALID_SESSION::Errors
        elseif s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/animeplanet_user" function animeplanet_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    user_data = retry(() -> get_animeplanet_user(username))
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)_count"] == 0
            continue
        end
        list = retry(() -> get_animeplanet_list(m, username))
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    userid_data = Dict([x => user_data[x] for x in ["username", "userid", "version"]]...)
    ret = Dict("user" => user_data, "items" => items, "userid" => userid_data)
    HTTP.Response(200, encode(ret, :json)...)
end

function get_animeplanet_user(username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        s = request(
            "animeplanet",
            Dict(
                "token" => token,
                "sessionid" => sessionid,
                "endpoint" => "user",
                "username" => username,
            ),
        )
        if s.status == 401
            invalidate_session(token)
            return INVALID_SESSION::Errors
        elseif s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_animeplanet_list(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        feed = let
            s = request(
                "animeplanet",
                Dict(
                    "token" => token,
                    "sessionid" => sessionid,
                    "endpoint" => "feed",
                    "username" => username,
                    "medium" => medium,
                ),
            )
            if s.status == 401
                invalidate_session(token)
                return INVALID_SESSION::Errors
            elseif s.status >= 400
                return NOT_FOUND::Errors
            end
            decode(s)
        end
        entries = []
        page = 1
        expand_pagelimit = false
        while true
            s = request(
                "animeplanet",
                Dict(
                    "token" => token,
                    "sessionid" => sessionid,
                    "endpoint" => "list",
                    "username" => username,
                    "medium" => medium,
                    "page" => page,
                    "expand_pagelimit" => expand_pagelimit,
                ),
            )
            if s.status == 401
                invalidate_session(token)
                return INVALID_SESSION::Errors
            elseif s.status >= 400
                return NOT_FOUND::Errors
            end
            data = decode(s)
            if get(data, "extend_pagelimit", false)
                expand_pagelimit = true
                continue
            end
            expand_pagelimit = false
            for x in data["entries"]
                x["updated_at"] = get(feed, x["itemid"], nothing)
            end
            append!(entries, data["entries"])
            if !data["next"]
                return entries
            end
            page += 1
        end
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/animeplanet_media" function animeplanet_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = retry(() -> get_animeplanet_media(medium, itemid))
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :json)...)
end

function get_animeplanet_media(medium::String, itemid::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAIABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        s = request(
            "animeplanet",
            Dict(
                "token" => token,
                "sessionid" => sessionid,
                "endpoint" => "media",
                "medium" => medium,
                "itemid" => itemid,
            ),
        )
        if s.status == 401
            invalidate_session(token)
            return INVALID_SESSION::Errors
        elseif s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.serveparallel(; host = "0.0.0.0", port = PORT, access_log = nothing, metrics=false, show_banner=false)