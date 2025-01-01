import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/database.jl")
include("layer2_connector.jl")

const PORT = parse(Int, ARGS[1])
const LAYER_2_URL = ARGS[2]
const LAYER_3_URL = ARGS[3]
const TOKEN_TIMEOUT = parse(Int, ARGS[4])

const KITSU_AUTH = Dict()
const KITSU_AUTH_LOCK = ReentrantLock()

function get_kitsu_auth()
    lock(KITSU_AUTH_LOCK) do
        if !isempty(KITSU_AUTH)
            if time() < KITSU_AUTH["expiry_time"]
                return KITSU_AUTH["token"]
            end
            empty!(KITSU_AUTH)
        end
        r = request(
            "resources",
            Dict("method" => "take", "location" => "kitsu", "timeout" => TOKEN_TIMEOUT),
        )
        if r.status >= 400
            return TOKEN_UNAVAILABLE::Errors
        end
        token = decode(r)
        try
            r = request("kitsu", Dict("token" => token, "endpoint" => "token"))
            if r.status >= 400
                return TOKEN_UNAVAILABLE::Errors
            end
            data = decode(r)
            for k in keys(data)
                KITSU_AUTH[k] = data[k]
            end
            return KITSU_AUTH["token"]
        finally
            request("resources", Dict("method" => "put", "token" => token))
        end
    end
end

function get_user_page(source::String, df::DataFrames.DataFrame)
    req_source = get(Dict("mal" => "malweb"), source, source)
    r = request(
        "resources",
        Dict("method" => "take", "location" => req_source, "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    if source == "mal"
        args = Dict("token" => token, "endpoint" => "user", "username" => only(df.username))
    elseif source == "anilist"
        args = Dict("token" => token, "endpoint" => "user", "userid" => only(df.userid))
    elseif source == "kitsu"
        auth = get_kitsu_auth()
        if auth isa Errors
            return auth
        end
        args = Dict(
            "token" => token,
            "auth" => auth,
            "endpoint" => "user",
            "userid" => only(df.userid),
        )
    else
        @assert false
    end
    try
        s = request(req_source, args)
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        return decode(s)
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function enumerate_list_pages(source::String, df::DataFrames.DataFrame)
    username = only(df.username)
    userid = only(df.userid)
    counts = Dict("manga" => only(df.manga_count), "anime" => only(df.anime_count))
    if source == "mal"
        return [
            Dict("username" => username, "offset" => x, "medium" => m) for m in
                                                                           keys(counts) for
            x = 0:1000:counts[m]-1
        ]
    elseif source == "anilist"
        return [
            Dict("userid" => userid, "chunk" => x, "medium" => m) for m in keys(counts) for
            x = 1:Int(ceil(counts[m] / 500))
        ]
    elseif source == "kitsu"
        auth = get_kitsu_auth()
        if auth isa Errors
            return auth
        end
        return [
            Dict("userid" => userid, "offset" => x, "auth" => auth) for
            x = 0:500:sum(values(counts))
        ]
    else
        @assert false
    end
end

function get_list_page(source::String, page::Dict)
    r = request(
        "resources",
        Dict("method" => "take", "location" => source, "timeout" => TOKEN_TIMEOUT),
    )
    if r.status >= 400
        return TOKEN_UNAVAILABLE::Errors
    end
    token = decode(r)
    try
        s = request(source, merge(Dict("token" => token, "endpoint" => "list"), page))
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        data = decode(s)
        return Dict(
            "entries" => data["entries"],
            "next" => get(data, "next", false) != false,
        )
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function get_user_df(source, username)
    with_db(:read, 3) do db
        query = """
            SELECT username,userid,anime_count,manga_count FROM users
            WHERE (source, lower(username)) = (\$1, lower(\$2))
            """
        stmt = db_prepare(db, query)
        DataFrames.DataFrame(LibPQ.execute(stmt, (source, username)))
    end
end

function get_user(source::String, username::String)
    df = @timeout 0.3 get_user_df(source, username)
    if df == :timeout || DataFrames.nrow(df) != 1
        return :error
    end
    d = Dict()
    d["user"] = Threads.@spawn @handle_errors get_user_page(source, df)
    d["items"] = []
    list_pages = enumerate_list_pages(source, df)
    if list_pages isa Errors
        return :error
    end
    for page in list_pages
        push!(d["items"], Threads.@spawn @handle_errors get_list_page(source, page))
    end
    r = Dict("user" => fetch(d["user"]), "items" => fetch.(d["items"]))
    return r
    if r["user"] isa Errors || any(isa.(r["items"], Errors))
        return :error
    end
    if any(x["next"] for x in r["items"])
        return :error
    end
    r["items"] = vcat([x["entries"] for x in r["items"]]...)
    if source in ["mal", "kitsu"]
        num_items = r["user"]["anime_count"] + r["user"]["manga_count"]
    elseif source == "anilist"
        num_items = r["user"]["animeCount"] + r["user"]["mangaCount"]
    else
        @assert false
    end
    if num_items != length(r["items"])
        return :error
    end
    r
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
    if source == "kitsu"
        auth = get_kitsu_auth()
        if auth isa Errors
            return auth
        end
    end
    token = decode(r)
    try
        args = Dict("token" => token, "endpoint" => "userid", "username" => username)
        if source == "kitsu"
            args["auth"] = auth
            args["key"] = "name"
        end
        s = request(source, args)
        if s.status >= 400
            return NOT_FOUND::Errors
        end
        data = decode(s)
        return data["userid"]
    finally
        request("resources", Dict("method" => "put", "token" => token))
    end
end

function proxy(query::String, params::Dict)
    r = request(LAYER_3_URL, query, params)
    if r.status >= 400
        return r = HTTP.Response(r.status, [])
    end
    HTTP.Response(200, Vector{UInt8}(r.body))
end

Oxygen.@post "/setup" function setup(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    source = data["source"]
    get_user_df(source, username) # check db connection
    HTTP.Response(200, [])
end

Oxygen.@post "/user" function user(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    username = data["username"]
    source = data["source"]
    if source == "animeplanet"
        return proxy("animeplanet_user_parallel", Dict("username" => username))
    else
        ret = get_user(source, username)
        if ret == :error
            if source in ["mal", "animeplanet"]
                args = Dict("username" => username)
            elseif source in ["anilist", "kitsu"]
                userid = get_userid(source, username)
                if userid isa Errors
                    return HTTP.Response(Int(userid), [])
                end
                args = Dict("userid" => userid)
            end
            return proxy("$(source)_user", args)
        end
    end
    HTTP.Response(200, encode(ret, :json)...)
end

Oxygen.serveparallel(; host = "0.0.0.0", port = PORT, access_log = nothing, metrics=false, show_banner=false)
