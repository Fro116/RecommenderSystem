module layer3

const PORT = parse(Int, ARGS[1])
const LAYER_2_URL = ARGS[2]
const TOKEN_TIMEOUT = parse(Int, ARGS[3])
const RETRIES = parse(Int, ARGS[4])

import Oxygen
import UUIDs

include("../julia_utils/http.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

function request(url::String, query::String, params::Dict)
    url = "$url/$query"
    delays = ExponentialBackOff(;
        n = RETRIES+1,
        first_delay = 1,
        max_delay = 1000,
        factor = 2.0,
        jitter = 0.1,
    )
    for delay in delays
        r = HTTP.post(url, encode(params, :msgpack)..., status_exception = false)
        if r.status in [400, 401]
            logerror("auth error for $url $params")
            return r
        end
        if r.status < 400 || r.status in [403, 404]
            # 403 -> list is private
            # 404 -> invalid user
            return r
        end
        if HTTP.hasheader(r.headers, "Retry-After")
            try
                retry_after = first(HTTP.headers(r, "Retry-After"))
                retry_timeout = min(parse(Int, retry_after), 1000)
                delay = max(delay, retry_timeout)
            catch
                logerror("invalid Retry-After $retry_after for $url $params")
            end
        end
        logerror("retrying $url $params after $delay seconds")
        sleep(delay)
    end
    logerror("could not retrieve $url $params")
    HTTP.Response(500, [])
end

request(query::String, params::Dict) = request(LAYER_2_URL, query, params)

@enum Errors begin
    INVALID_SESSION = 401
    NOT_FOUND = 404
    TOKEN_UNAVAILABLE = 503
end

macro retry(f, retries=3)
    quote
        begin
            local x = $(esc(f))
            for _ in 1:$(esc(retries))-1
                if x ∉ [INVALID_SESSION, TOKEN_UNAVAILABLE]
                    break
                end
                logerror("retrying function after error $x")
                sleep(1)
                x = $(esc(f))
            end
            x
        end
    end
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
            return TOKEN_UNAVAILABLE::Errors
        end
        data = decode(r)
        sessionid = data["token"]
    elseif token["resource"]["location"] == "animeplanet"
        sessionid = string(UUIDs.uuid4())
        r = request(
            "animeplanet",
            Dict("token" => token, "sessionid" => sessionid, "endpoint" => "login")
        )
        if r.status >= 400
            logerror("animeplanet login failed with $token")
            return TOKEN_UNAVAILABLE::Errors
        end
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

function fetch_user_parallel(source, user_data, items)
    ret = Dict()
    user_data = fetch(user_data)
    if user_data isa Errors
        return HTTP.Response(Int(user_data), [])
    end
    ret["user"] = user_data
    ret["items"] = []
    for t in items
        x = fetch(t)
        if x isa Errors
            return HTTP.Response(Int(x), [])
        end
        append!(ret["items"], x)
    end
    if source in ["mal", "anilist", "animeplanet"]
        ret["usermap"] = Dict([x => user_data[x] for x in ["username", "userid"]]...)
    elseif source in ["kitsu"]
        ret["usermap"] = Dict(
            "username" => user_data["name"],
            "userid" => user_data["userid"],
        )
    else
        @assert false
    end
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/mal_username" function mal_username(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    data = @retry get_malweb_username(userid)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_malweb_username(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "malweb", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/mal_image" function mal_image(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    url = data["url"]
    data = @retry get_malweb_image(url)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_malweb_image(url::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "malweb", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "malweb",
            Dict("token" => token, "endpoint" => "image", "url" => url),
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
    user_data = @retry get_malweb_user(username)
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)_count"] == 0
            continue
        end
        list = @retry get_mal_list(m, username)
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    userid_data = Dict([x => user_data[x] for x in ["username", "userid", "version"]]...)
    ret = Dict("user" => user_data, "items" => items, "userid" => userid_data)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/mal_user_parallel" function mal_user_parallel(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    user_data = Threads.@spawn @retry get_malweb_user(username)
    items = []
    for m in ["manga", "anime"]
        list = Threads.@spawn @retry get_mal_list(m, username)
        push!(items, list)
    end
    fetch_user_parallel("mal", user_data, items)
end

function get_malweb_user(username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "malweb", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/mal_fingerprint" function mal_fingerprint(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    tasks = [Threads.@spawn @retry get_mal_fingerprint(m, username) for m in ["manga", "anime"]]
    ret = []
    for task in tasks
        t = fetch(task)
        if t isa Errors
            return HTTP.Response(Int(t), [])
        end
        append!(ret, t["entries"])
    end
    HTTP.Response(200, encode(Dict("fingerprint" => JSON3.write(ret)), :msgpack)...)
end

function get_mal_fingerprint(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "mal", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "mal",
            Dict(
                "token" => token,
                "endpoint" => "fingerprint",
                "username" => username,
                "medium" => medium,
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

Oxygen.@post "/mal_media" function mal_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    details = @retry get_mal_media("mal", medium, itemid)
    if isa(details, Errors)
        return HTTP.Response(Int(details), [])
    end
    relations = @retry get_mal_media("malweb", medium, itemid)
    if isa(relations, Errors)
        return HTTP.Response(Int(relations), [])
    end
    HTTP.Response(200, encode(merge(details, relations), :msgpack)...)
end

function get_mal_media(location::String, medium::String, itemid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => location, "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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
    user_data = @retry get_anilist_user(userid)
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)Count"] == 0
            continue
        end
        list = @retry get_anilist_list(m, userid)
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    ret = Dict("user" => user_data, "items" => items)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/anilist_user_parallel" function anilist_user_parallel(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    user_data = Threads.@spawn @retry get_anilist_user(userid)
    items = []
    for m in ["manga", "anime"]
        list = Threads.@spawn @retry get_anilist_list(m, userid)
        push!(items, list)
    end
    fetch_user_parallel("anilist", user_data, items)
end

function get_anilist_user(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

function get_anilist_list(medium::String, userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/anilist_fingerprint" function anilist_fingerprint(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    tasks = [Threads.@spawn @retry get_anilist_fingerprint(m, userid) for m in ["manga", "anime"]]
    ret = []
    for task in tasks
        t = fetch(task)
        if t isa Errors
            return HTTP.Response(Int(t), [])
        end
        append!(ret, t["entries"])
    end
    HTTP.Response(200, encode(Dict("fingerprint" => JSON3.write(ret)), :msgpack)...)
end

function get_anilist_fingerprint(medium::String, userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "anilist",
            Dict(
                "token" => token,
                "endpoint" => "fingerprint",
                "userid" => userid,
                "medium" => medium,
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

Oxygen.@post "/anilist_media" function anilist_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = @retry get_anilist_media(medium, itemid)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_anilist_media(medium::String, itemid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/anilist_image" function anilist_image(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    url = data["url"]
    data = @retry get_anilist_image(url)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_anilist_image(url::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "anilist", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        s = request(
            "anilist",
            Dict(
                "token" => token,
                "endpoint" => "image",
                "url" => url,
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

function get_userid(source::String, username::String)
    @assert source in ["anilist", "kitsu"]
    r = request(
        "resources",
        Dict("method" => "take", "location" => source, "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    args = Dict("token" => token, "endpoint" => "userid", "username" => username)
    if source == "kitsu"
        auth = get_session(token)
        if isa(auth, Errors)
            return auth
        end
        if auth isa Errors
            return auth
        end
        args["auth"] = auth
        args["key"] = "name"
    end
    try
        s = request(source, args)
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/anilist_userid" function anilist_userid(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    data = @retry get_userid("anilist", username)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

Oxygen.@post "/kitsu_user" function kitsu_user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    user_data = @retry get_kitsu_user(userid)
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    if (user_data["manga_count"] + user_data["anime_count"]) > 0
        list = @retry get_kitsu_list(userid)
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    ret = Dict("user" => user_data, "items" => items)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/kitsu_user_parallel" function kitsu_user_parallel(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    user_data = Threads.@spawn @retry get_kitsu_user(userid)
    items = [Threads.@spawn @retry get_kitsu_list(userid)]
    fetch_user_parallel("kitsu", user_data, items)
end

function get_kitsu_user(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

function get_kitsu_list(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/kitsu_fingerprint" function kitsu_fingerprint(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    userid = data["userid"]
    tasks = [Threads.@spawn @retry get_kitsu_fingerprint(userid)]
    ret = []
    for task in tasks
        t = fetch(task)
        if t isa Errors
            return HTTP.Response(Int(t), [])
        end
        append!(ret, t["entries"])
    end
    HTTP.Response(200, encode(Dict("fingerprint" => JSON3.write(ret)), :msgpack)...)
end

function get_kitsu_fingerprint(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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
                "endpoint" => "fingerprint",
                "auth" => auth,
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

Oxygen.@post "/kitsu_media" function kitsu_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = @retry get_kitsu_media(medium, itemid)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_kitsu_media(medium::String, itemid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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

Oxygen.@post "/kitsu_userid" function kitsu_userid(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    data = @retry get_userid("kitsu", username)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

Oxygen.@post "/kitsu_image" function kitsu_image(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    url = data["url"]
    data = @retry get_kitsu_image(url)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_kitsu_image(url::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
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
                "endpoint" => "image",
                "url" => url,
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
    data = @retry get_animeplanet_username(userid)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_animeplanet_username(userid::Integer)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
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
    user_data = @retry get_animeplanet_user(username)
    if isa(user_data, Errors)
        return HTTP.Response(Int(user_data), [])
    end
    items = []
    for m in ["manga", "anime"]
        if user_data["$(m)_count"] == 0
            continue
        end
        list = @retry get_animeplanet_list(m, username, parallel=false)
        if isa(list, Errors)
            return HTTP.Response(Int(list), [])
        end
        append!(items, list)
    end
    userid_data = Dict([x => user_data[x] for x in ["username", "userid", "version"]]...)
    ret = Dict("user" => user_data, "items" => items, "userid" => userid_data)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/animeplanet_user_parallel" function animeplanet_user_parallel(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    user_data = Threads.@spawn @retry get_animeplanet_user(username)
    items = []
    for m in ["manga", "anime"]
        push!(items, Threads.@spawn @retry get_animeplanet_list(m, username, parallel=true))
    end
    fetch_user_parallel("animeplanet", user_data, items)
end

function get_animeplanet_user(username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
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

function get_animeplanet_list(medium::String, username::String; parallel::Bool)
    if parallel
        feed_task = Threads.@spawn @handle_errors get_animeplanet_feed(medium, username)
        entries_task = Threads.@spawn @handle_errors get_animeplanet_entries(medium, username)
        feed = fetch(feed_task)
        entries = fetch(entries_task)
    else
        feed = get_animeplanet_feed(medium, username)
        if feed isa Errors
            return feed
        end
        entries = get_animeplanet_entries(medium, username)
    end
    for x in [feed, entries]
        if x isa Errors
            return x
        end
    end
    for x in entries
        x["updated_at"] = get(feed, x["itemid"], nothing)
    end
    entries
end

function get_animeplanet_feed(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
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
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_animeplanet_entries(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
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
            expand_pagelimit = get(data, "expand_pagelimit", false)
            append!(entries, get(data, "entries", []))
            if "nextpage" in keys(data)
                page = data["nextpage"]
            else
                return entries
            end
        end
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/animeplanet_fingerprint" function animeplanet_fingerprint(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    tasks = [Threads.@spawn @retry get_animeplanet_fingerprint(m, username) for m in ["manga", "anime"]]
    ret = []
    for task in tasks
        t = fetch(task)
        if t isa Errors
            return HTTP.Response(Int(t), [])
        end
        append!(ret, t["entries"])
    end
    HTTP.Response(200, encode(Dict("fingerprint" => JSON3.write(ret)), :msgpack)...)
end

function get_animeplanet_fingerprint(medium::String, username::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
        s = request(
            "animeplanet",
            Dict(
                "token" => token,
                "sessionid" => sessionid,
                "endpoint" => "fingerprint",
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
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

Oxygen.@post "/animeplanet_media" function animeplanet_media(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    medium = data["medium"]
    itemid = data["itemid"]
    data = @retry get_animeplanet_media(medium, itemid)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_animeplanet_media(medium::String, itemid::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
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

Oxygen.@post "/animeplanet_image" function animeplanet_image(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    url = data["url"]
    data = @retry get_animeplanet_image(url)
    if isa(data, Errors)
        return HTTP.Response(Int(data), [])
    end
    HTTP.Response(200, encode(data, :msgpack)...)
end

function get_animeplanet_image(url::String)
    r = request(
        "resources",
        Dict("method" => "take", "location" => "animeplanet", "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        sessionid = get_session(token)
        if sessionid isa Errors
            return sessionid
        end
        s = request(
            "animeplanet",
            Dict(
                "token" => token,
                "sessionid" => sessionid,
                "endpoint" => "image",
                "url" => url,
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

function compile(::Integer) end
include("../julia_utils/start_oxygen.jl")

end
