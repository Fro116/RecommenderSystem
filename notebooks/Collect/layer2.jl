module layer2

const PORT = parse(Int, ARGS[1])
const RATELIMIT_WINDOW = parse(Int, ARGS[2])
const LAYER_1_URLS = split(ARGS[3], ",")
const DEFAULT_IMPERSONATE = parse(Bool, ARGS[4])
const DEFAULT_TIMEOUT = parse(Int, ARGS[5])
const USE_SHARED_IPS = parse(Bool, ARGS[6])
const API_VERSION = "5.3.0"

import CSV
import DataFrames
import Dates
import Glob
import HTTP
import JSON3
import Memoize: @memoize
import Oxygen
import Random
import UUIDs

include("../julia_utils/hash.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/stdout.jl")

function logstatus(fn, r, url; handled_errors = [403, 404])
    if r.status ∉ handled_errors
        logerror("$fn received error code $(r.status) for $url")
    end
end

const Resource = Dict

function load_resources()::Vector{Resource}
    credentials = Dict()
    function get_proxies(path)
        proxies = []
        proxy_df = CSV.read(
            path,
            DataFrames.DataFrame,
        )
        for (host, port, username, password) in
            zip(proxy_df.host, proxy_df.port, proxy_df.username, proxy_df.password)
            ip = split(username, "-")[end]
            push!(proxies, "http://$username:$password@$host:$port")
        end
        Random.shuffle(proxies)
    end
    dedicated = get_proxies("../../secrets/ips.dedicated.txt")
    if USE_SHARED_IPS
        shared = get_proxies("../../secrets/ips.shared.txt")
        mal_tokens = readlines("../../secrets/mal.auth.shared.txt")
    else
        shared = dedicated
        mal_tokens = readlines("../../secrets/mal.auth.dedicated.txt")
    end
    mal_resources = [
        Dict("location" => "mal", "token" => x, "proxyurls" => [], "ratelimit" => 8) for
        x in mal_tokens
    ]
    i = 1
    for proxy in shared
        if length(mal_resources[i]["proxyurls"]) < 10
            push!(mal_resources[i]["proxyurls"], proxy)
        end
        i = (i % length(mal_resources)) + 1
    end
    mal_resources = [x for x in mal_resources if !isempty(x["proxyurls"])]

    malweb_resources =
        [Dict("location" => "malweb", "proxyurl" => x, "ratelimit" => 8) for x in shared]

    anilist_resources =
        [Dict("location" => "anilist", "proxyurl" => x, "ratelimit" => 8) for x in shared]

    kitsu_credentials = []
    for x in readlines("../../secrets/kitsu.auth.txt")
        username, password = split(x, ",")
        push!(kitsu_credentials, Dict("username" => username, "password" => password))
    end
    kitsu_resources = [
        Dict(
            "location" => "kitsu",
            "credentials" => x,
            "proxyurls" => [],
            "ratelimit" => 8,
        ) for x in kitsu_credentials
    ]
    i = 1
    for proxy in shared
        if length(kitsu_resources[i]["proxyurls"]) < 10
            push!(kitsu_resources[i]["proxyurls"], proxy)
        end
        i = (i % length(kitsu_resources)) + 1
    end
    kitsu_resources = [x for x in kitsu_resources if !isempty(x["proxyurls"])]

    animeplanet_credentials = []
    for x in readlines("../../secrets/animeplanet.auth.txt")
        username, password = split(x, ",")
        push!(animeplanet_credentials, Dict("username" => username, "password" => password))
    end
    animeplanet_resources = [
        Dict(
            "location" => "animeplanet",
            "proxyurl" => x,
            "ratelimit" => 8,
            "credentials" => animeplanet_credentials[(i % length(animeplanet_credentials))+1],
        ) for (i, x) in Iterators.enumerate(dedicated)
    ]

    resources = vcat(
        mal_resources,
        malweb_resources,
        anilist_resources,
        kitsu_resources,
        animeplanet_resources,
    )
    Random.shuffle(resources)
end

struct ResourceMetadata
    version::Int
    checkout_time::Union{Nothing,Float64}
    request_times::Vector{Float64}
end

mutable struct Resources
    resources::Dict{Resource,ResourceMetadata} # resource -> (version, checkout time)
    index::Dict{String,Vector{Resource}} # location -> [resources]
    queue::Dict{String,Vector{String}} # location -> [task uuid]
    lock::ReentrantLock
end

function Resources(resources)
    index = Dict{String,Vector{Resource}}()
    for x in resources
        loc = x["location"]
        if loc ∉ keys(index)
            index[loc] = []
        end
        push!(index[loc], x)
    end
    Resources(
        Dict(x => ResourceMetadata(0, nothing, []) for x in resources),
        index,
        Dict(x => [] for x in keys(index)),
        ReentrantLock(),
    )
end

const RESOURCES = Resources(load_resources());
const RECLAIM_TIMEOUT = 1000
LAST_RECLAIM::Float64 = time()

function reclaim(r::Resources)
    lock(r.lock) do
        t = time()
        global LAST_RECLAIM
        LAST_RECLAIM = t
        for loc in keys(r.index)
            delta = 0
            li = length(r.index[loc])
            lq = length(r.queue[loc])
            if li > 0
                k = first(r.index[loc])
                m = r.resources[k]
                delta = ratelimit_delta(m, k["ratelimit"], RATELIMIT_WINDOW)
            else
                delta = 0
            end
            logtag("RESOURCES", "location $loc has $li resources and $lq tasks with a wait delay of $delta")
        end
        for k in Set(keys(r.resources))
            m = r.resources[k]
            if !isnothing(m.checkout_time) && t - m.checkout_time > RECLAIM_TIMEOUT
                r.resources[k] =
                    ResourceMetadata(m.version + 1, nothing, m.request_times)
                push_front!(r.index[k["location"]], k)
            end
        end
    end
end

function take!(r::Resources, location::String, timeout::Real)
    start = time()
    uuid = string(UUIDs.uuid4())
    lock(r.lock) do
        if start - LAST_RECLAIM > RECLAIM_TIMEOUT
            reclaim(r)
        end
        push!(r.queue[location], uuid)
    end
    try
        while true
            val = lock(r.lock) do
                task_id = first(r.queue[location])
                if task_id != uuid
                    return nothing
                end
                if isempty(r.index[location])
                    return nothing
                end
                if timeout == 0
                    k = first(r.index[location])
                    m = r.resources[k]
                    if ratelimit_delta(m, k["ratelimit"], RATELIMIT_WINDOW) > 0
                        return nothing
                    end
                end
                k = popfirst!(r.index[location])
                m = r.resources[k]
                @assert isnothing(m.checkout_time)
                r.resources[k] = ResourceMetadata(m.version, time(), m.request_times)
                Dict("resource" => k, "version" => m.version)
            end
            if !isnothing(val)
                return val
            end
            if time() - start > timeout
                return nothing
            end
            sleep(0.01)
        end
    finally
        lock(r.lock) do
            arr = r.queue[location]
            deleteat!(arr, findfirst(x -> x == uuid, arr))
        end
    end
end

function put!(r::Resources, resource::Dict, version::Integer)
    lock(r.lock) do
        if resource ∉ keys(r.resources)
            return
        end
        m = r.resources[resource]
        if m.version != version
            return
        end
        r.resources[resource] = ResourceMetadata(m.version, nothing, m.request_times)
        push!(r.index[resource["location"]], resource)
    end
end

Oxygen.@post "/resources" function resources_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    if data["method"] == "take"
        token = take!(RESOURCES, data["location"], data["timeout"])
        if isnothing(token)
            return HTTP.Response(404, [])
        end
        return HTTP.Response(200, encode(token, :msgpack)...)
    elseif data["method"] == "put"
        put!(RESOURCES, data["token"]["resource"], data["token"]["version"])
        return HTTP.Response(200, [])
    else
        @assert false
    end
end

function ratelimit_delta(x::ResourceMetadata, ratelimit::Real, window::Integer)
    if length(x.request_times) >= window
        startindex = length(x.request_times) - window + 1
        times = x.request_times[startindex:end]
        wait_until = first(times) + length(times) * ratelimit
        delta = wait_until - time()
        if delta > 0
            return delta
        end
    end
    0
end

function ratelimit!(x::ResourceMetadata, ratelimit::Real)
    delay = ratelimit_delta(x, ratelimit, RATELIMIT_WINDOW)
    if delay > 0
        sleep(delay)
    end
    push!(x.request_times, time())
    if length(x.request_times) > RATELIMIT_WINDOW
        popfirst!(x.request_times)
    end
end

struct Response
    status::Integer
    body::String
    headers::Dict{String,String}
end

function HTTP.Response(x::Response)
    HTTP.Response(x.status, HTTP.Headers(collect(x.headers)), Vector{UInt8}(x.body))
end


function callproxy(
    method::String,
    url::String,
    headers::Dict{String,<:Any},
    body::Union{Vector{UInt8},Nothing},
    proxyurl::Union{String,Nothing},
    sessionid::String,
)
    args = Dict{String,Any}(
        "method" => method,
        "url" => url,
        "sessionid" => sessionid,
        "timeout" => DEFAULT_TIMEOUT
    )
    if get(headers, "impersonate", DEFAULT_IMPERSONATE)
        args["impersonate"] = "chrome"
        delete!(headers, "impersonate")
    end
    if !isnothing(body)
        @assert headers["Content-Type"] == "application/json"
        delete!(headers, "Content-Type")
        args["json"] = String(body)
    end
    if !isempty(headers)
        args["headers"] = Dict{String,String}(headers)
    end
    if !isnothing(proxyurl)
        args["proxyurl"] = proxyurl
    end
    layer_1_url = LAYER_1_URLS[(shahash(sessionid) % length(LAYER_1_URLS)) + 1]
    r = HTTP.post(layer_1_url, encode(args, :json)..., status_exception = false)
    Response(r.status, String(r.body), Dict(k => v for (k, v) in r.headers))
end

function request(
    resource::Resource,
    method::String,
    url::String,
    headers::Dict{String,<:Any} = Dict(),
    body::Union{Vector{UInt8},Nothing} = nothing,
)::Response
    metadata = lock(RESOURCES.lock) do
        m = get(RESOURCES.resources, resource, nothing)
        if isnothing(m)
            return Response(500, "", Dict())
        end
        RESOURCES.resources[resource] =
            ResourceMetadata(m.version, time(), m.request_times)
    end
    ratelimit!(metadata, resource["ratelimit"])
    if "proxyurls" in keys(resource)
        proxyurl = rand(resource["proxyurls"])
    else
        proxyurl = resource["proxyurl"]
    end
    sessionid = string(shahash((resource, proxyurl, Dates.today())))
    if "sessionid" in keys(headers)
        sessionid = pop!(headers, "sessionid")
    end
    callproxy(method, url, headers, body, proxyurl, sessionid)
end

@memoize function html_entity_map()
    Dict(
        String(k) => v["characters"] for (k, v) in JSON3.read(read("entities.json", String))
    )
end

html_unescape(::Nothing) = nothing
function html_unescape(text::AbstractString)
    entities = Dict(k => v for (k, v) in html_entity_map() if occursin(k, text))
    # greedy match replacements
    for k in sort(collect(keys(entities)), by = length, rev = true)
        text = replace(text, k => entities[k])
    end
    text = replace(text, entities...) # html entities
    try
        text = replace(
            text,
            [
                x.match => Char(parse(Int, only(x.captures))) for
                x in eachmatch(r"&#(\d+);", text)
            ]...,
        ) # numeric entities
        text = replace(
            text,
            [
                x.match => Char(parse(Int, only(x.captures), base = 16)) for
                x in eachmatch(r"&#x([0-9a-fA-F]+);", text)
            ]...,
        ) # hex entities
    catch
        logerror("html_unescape could not parse $text")
    end
    text
end

function extract(
    text,
    start,
    stop;
    capture = """(?s)(.*?)""",
    optional = false,
    multiple = false,
)
    regex = Regex(start * capture * stop)
    matches = [only(m.captures) for m in eachmatch(regex, text)]
    matches = [x for (i, x) in enumerate(matches) if findfirst(==(x), matches) == i]
    if optional && isempty(matches)
        return nothing
    end
    if multiple
        return [strip(x) for x in matches]
    end
    strip(only(matches))
end

optget(x::AbstractDict, k::String) = get(x, k, nothing)

Oxygen.@post "/mal" function mal_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    if resource["location"] != "mal"
        logerror("""mal_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "list"
            return mal_get_list(resource, data["username"], data["medium"], data["offset"])
        elseif endpoint == "fingerprint"
            return mal_get_fingerprint(resource, data["username"], data["medium"])
        elseif endpoint == "media"
            return mal_get_media(resource, data["medium"], data["itemid"])
        else
            logerror("mal_api invalid endpoint $endpoint")
            return HTTP.Response(500, [])
        end
    catch e
        args = Dict(k => v for (k, v) in data if k != "token")
        logerror("mal_api error $e for $args")
        return HTTP.Response(500, [])
    end
end

function mal_get_list(resource::Resource, username::String, medium::String, offset::Integer)
    if medium == "anime"
        progress_col = "num_episodes_watched"
        repeat_col = "is_rewatching"
        repeat_count_col = "num_times_rewatched"
        repeat_value_col = "rewatch_value"
    elseif medium == "manga"
        progress_col = "num_chapters_read"
        repeat_col = "is_rereading"
        repeat_count_col = "num_times_reread"
        repeat_value_col = "reread_value"
    else
        @assert false
    end
    params = Dict("limit" => 1000, "fields" => "list_status", "nsfw" => true)
    if offset != 0
        params["offset"] = offset
    end
    url = string(
        HTTP.URI(
            "https://api.myanimelist.net/v2/users/$username/$(medium)list";
            query = params,
        ),
    )
    entries = []
    headers = Dict("X-MAL-CLIENT-ID" => resource["token"])
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("mal_get_list", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    if "data" ∉ keys(json)
        logerror("mal_get_list received empty json $(keys(json)) for $url")
        return HTTP.Response(500, [])
    end
    for x in json["data"]
        ls = x["list_status"]
        d = Dict(
            "version" => API_VERSION,
            "username" => username,
            "medium" => medium,
            "itemid" => x["node"]["id"],
            "status" => optget(ls, "status"),
            "score" => optget(ls, "score"),
            "progress" => optget(ls, progress_col),
            "num_volumes_read" => optget(ls, "num_volumes_read"),
            "start_date" => optget(ls, "start_date"),
            "finish_date" => optget(ls, "finish_date"),
            "priority" => optget(ls, "priority"),
            "repeat_col" => optget(ls, repeat_col),
            "repeat_count" => optget(ls, repeat_count_col),
            "repeat_value" => optget(ls, repeat_value_col),
            "tags" => optget(ls, "tags"),
            "comments" => optget(ls, "comments"),
            "updated_at" => optget(ls, "updated_at"),
        )
        push!(entries, d)
    end
    ret = Dict("entries" => entries, "offset" => offset, "limit" => params["limit"])
    if "next" in keys(json["paging"])
        ret["next"] = json["paging"]["next"]
    end
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function mal_get_fingerprint(resource::Resource, username::String, medium::String)
    params = Dict("limit" => 1, "fields" => "list_status{updated_at}", "nsfw" => true, "sort" => "list_updated_at")
    url = string(
        HTTP.URI(
            "https://api.myanimelist.net/v2/users/$username/$(medium)list";
            query = params,
        ),
    )
    entries = []
    headers = Dict("X-MAL-CLIENT-ID" => resource["token"])
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("mal_get_fingerprint", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    if "data" ∉ keys(json)
        logerror("mal_get_fingerprint received empty json $(keys(json)) for $url")
        return HTTP.Response(500, [])
    end
    for x in json["data"]
        ls = x["list_status"]
        d = Dict(
            "version" => API_VERSION,
            "medium" => medium,
            "itemid" => x["node"]["id"],
            "updated_at" => optget(ls, "updated_at"),
        )
        push!(entries, d)
    end
    ret = Dict("entries" => entries)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function mal_get_media(resource::Resource, medium::String, itemid::Integer)
    fields = Dict(
        "common" => [
            "id",
            "title",
            "main_picture",
            "alternative_titles",
            "start_date",
            "end_date",
            "synopsis",
            "nsfw",
            "genres",
            "created_at",
            "updated_at",
            "media_type",
            "pictures",
            "background",
            "recommendations",
            "status",
        ],
        "anime" => [
            "num_episodes",
            "start_season",
            "broadcast",
            "source",
            "average_episode_duration",
            "rating",
            "studios",
        ],
        "manga" => ["num_volumes", "num_chapters", "authors{first_name,last_name}", "serialization{name}"],
    )
    params = Dict("fields" => join(vcat(fields["common"], fields[medium]), ","))
    url = string(HTTP.URI("https://api.myanimelist.net/v2/$medium/$itemid"; query = params))
    entries = []
    headers = Dict("X-MAL-CLIENT-ID" => resource["token"])
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("mal_get_media", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    details = Dict(
        "version" => API_VERSION,
        "medium" => medium,
        "itemid" => json["id"],
        "title" => json["title"],
        "alternative_titles" => json["alternative_titles"],
        "start_date" => optget(json, "start_date"),
        "end_date" => optget(json, "end_date"),
        "synopsis" => optget(json, "synopsis"),
        "genres" =>  optget(json, "genres"),
        "created_at" => optget(json, "created_at"),
        "updated_at" => optget(json, "updated_at"),
        "media_type" => json["media_type"],
        "nsfw" => json["nsfw"],
        "pictures" => json["pictures"],
        "background" => json["background"],
        "recommendations" => [
            Dict(
                "medium" => medium,
                "itemid" => x["node"]["id"],
                "count" => x["num_recommendations"],
            ) for x in json["recommendations"]
        ],
        "status" => optget(json, "status"),
        "num_episodes" => optget(json, "num_episodes"),
        "start_season" => optget(json, "start_season"),
        "broadcast" => optget(json, "broadcast"),
        "source" => optget(json, "source"),
        "average_episode_duration" => optget(json, "average_episode_duration"),
        "rating" => optget(json, "rating"),
        "studios" =>
            "studios" in keys(json) ? [x["name"] for x in json["studios"]] :
            "serialization" in keys(json) ? [x["node"]["name"] for x in json["serialization"]] :
            nothing,
        "num_volumes" => optget(json, "num_volumes"),
        "num_chapters" => optget(json, "num_chapters"),
        "authors" =>
            "authors" in keys(json) ?
            [Dict("first_name" => x["node"]["first_name"], "last_name" => x["node"]["last_name"], "role" => x["role"]) for x in json["authors"]] : nothing,
    )
    # the mal API does not return manga relations for anime entries and vice versa        
    ret = Dict("details" => details)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/malweb" function malweb_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    if resource["location"] != "malweb"
        logerror("""malweb_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "media"
            return malweb_get_media(resource, data["medium"], data["itemid"])
        elseif endpoint == "username"
            return malweb_get_username(resource, data["userid"])
        elseif endpoint == "user"
            return malweb_get_user(resource, data["username"])
        elseif endpoint == "image"
            return malweb_get_image(resource, data["url"])
        else
            logerror("malweb_api invalid endpoint $endpoint")
            return HTTP.Response(500, [])
        end
    catch e
        args = Dict(k => v for (k, v) in data if k != "token")
        logerror("malweb_api error $e for $args")
        return HTTP.Response(500, [])
    end
end

function malweb_get_username(resource::Resource, userid::Integer)
    url = "https://myanimelist.net/comments.php?id=$userid"
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("malweb_get_username", r, url)
        return HTTP.Response(r)
    end
    for m in eachmatch(r"/profile/([^\"/%]+)\"", r.body)
        username = only(m.captures)
        ret = Dict("version" => API_VERSION, "userid" => userid, "username" => username)
        return HTTP.Response(200, encode(ret, :msgpack)...)
    end
    HTTP.Response(404, [])
end

function malweb_get_media(resource::Resource, medium::String, itemid::Integer)
    ret = Dict()
    url = "https://myanimelist.net/$medium/$itemid"
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("malweb_get_media", r, url)
        return HTTP.Response(r)
    end
    ret["relations"] = malweb_get_media_relations(r.body, medium, itemid)
    url = "https://myanimelist.net/$medium/$itemid/$itemid/userrecs"
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("malweb_get_media", r, url)
        return HTTP.Response(r)
    end
    ret["recommendations"] = malweb_get_media_recommendations(r.body, medium, itemid)
    reviews = []
    reviews_page = 0
    next_page = true
    while next_page
        reviews_page += 1
        url = "https://myanimelist.net/$medium/$itemid/$itemid/reviews?spoiler=on&p=$reviews_page"
        r = request(resource, "GET", url, Dict("impersonate" => true))
        if r.status >= 400
            logstatus("malweb_get_media", r, url)
            return HTTP.Response(r)
        end
        revs, next_page = malweb_get_media_reviews(r.body, medium, itemid)
        append!(reviews, revs)
    end
    ret["reviews"] = reviews
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function malweb_get_media_relations(text::String, medium::String, itemid::Integer)
    relation_types = Set([
        "Sequel",
        "Prequel",
        "Alternative Setting",
        "Alternative Version",
        "Side Story",
        "Summary",
        "Full Story",
        "Parent Story",
        "Spin-Off",
        "Adaptation",
        "Character",
        "Other",
    ])
    records = Set()
    related_entries_section = false
    last_line = nothing
    last_relation = nothing
    last_href = nothing
    picture_section = true
    for match in eachmatch(r"([^<>]+|</?[^>]+>)", text)
        line = strip(match.match)
        prev_line = last_line
        cur_line = line
        last_line = line
        if line == "Related Entries"
            related_entries_section = true
            continue
        end
        if !related_entries_section
            continue
        end
        if line == """<td class="pb24">"""
            if !isnothing(last_href)
                logerror(
                    "malweb_get_media_relations $medium $itemid did not finish parsing $last_href",
                )
            end
            return collect(records)
        end
        if prev_line == """<div class="relation">"""
            line = strip(first(split(line, "\n")))
            last_relation = line
            if last_relation ∉ relation_types
                logerror(
                    "malweb_get_media_relations $medium $itemid could not parse relation $line",
                )
                continue
            end
            continue
        end
        if prev_line == """<td valign="top" class="ar fw-n borderClass nowrap">"""
            picture_section = false
            line = line[1:end-1] # strip trailing colon
            last_relation = line
            if last_relation ∉ relation_types
                logerror(
                    "malweb_get_media_relations $medium $itemid could not parse relation $line",
                )
                continue
            end
            continue
        end
        for m in
            eachmatch(r"""<a href="https://myanimelist.net/(manga|anime)/([0-9]+)/""", line)
            if picture_section
                if isnothing(last_href)
                    last_href = line
                    continue
                elseif last_href == line
                    last_href = nothing
                else
                    logerror(
                        "malweb_get_media_relations $medium $itemid unexpected href $line",
                    )
                    last_href = nothing
                    continue
                end
            end
            if isnothing(last_relation)
                logerror(
                    "malweb_get_media_relations $medium $itemid could not find relation for $line",
                )
                continue
            end
            m_medium, m_itemid = m.captures
            d = Dict(
                "version" => API_VERSION,
                "relation" => last_relation,
                "itemid" => itemid,
                "medium" => medium,
                "target_id" => parse(Int, m_itemid),
                "target_medium" => m_medium,
            )
            push!(records, d)
            continue
        end
    end
    if related_entries_section
        logerror("malweb_get_media_relations $medium $itemid could not parse relations")
    end
    collect(records)
end

function malweb_get_media_recommendations(text::String, medium::String, itemid::Integer)
    all_recommendations = []
    item_chunks = split(
        text,
        r"""<div class="picSurround"><a href="https://myanimelist.net/(?:anime|manga)/""",
    )
    if length(item_chunks) < 2
        return []
    end
    user_rec_regex =
        r"(?s)class=\"spaceit_pad detail-user-recs-text\"[^>]*>(.*?)</div>.*?Recommended by <a href=\"/profile/([^\"]+)\">"
    for chunk in item_chunks[2:end]
        rec_itemid_match = match(r"^([0-9]+)/", chunk)
        if isnothing(rec_itemid_match)
            continue
        end
        rec_itemid = parse(Int, rec_itemid_match.captures[1])
        for user_match in eachmatch(user_rec_regex, chunk)
            raw_text, username = user_match.captures
            final_text =
                replace(
                    raw_text,
                    r"<span style=\"display: none;\"[^>]*>" => "",
                    "</span>" => "",
                    r"<a [^>]*?>read more</a>" => "",
                    "&nbsp;" => " ",
                    r"<br\s*/?>" => "\n",
                ) |>
                html_unescape |>
                (s -> replace(s, r"<[^>]*>" => "")) |>
                strip
            push!(
                all_recommendations,
                Dict("itemid" => rec_itemid, "username" => username, "text" => final_text),
            )
        end
    end
    all_recommendations
end

function malweb_get_media_reviews(text::String, medium::String, itemid::Integer)
    entries = []
    review_block_regex =
        r"(?s)<div class=\"review-element js-review-element\".*?</div>\s*</div>\s*</div>"
    for m in eachmatch(review_block_regex, text)
        block = m.match
        username_match = match(r"class=\"username\">\s*<a\s.*?>(.*?)</a>", block)
        text_match = match(r"(?s)<div class=\"text\">(.*?)</div>", block)
        if isnothing(username_match) || isnothing(text_match)
            continue
        end
        username = strip(only(username_match.captures))
        raw_text = only(text_match.captures)
        final_text =
            replace(
                raw_text,
                r"<span class=\"js-visible\".*?</span>"s => "",
                "<span class=\"js-hidden\" style=\"display: none;\">" => "",
                "</span>" => "",
                r"<br\s*\/?>" => "\n",
            ) |>
            html_unescape |>
            (s -> replace(s, r"<[^>]*>" => "")) |>
            strip
        rating_match = match(r"Reviewer’s Rating: <span class=\"num\">(\d+)</span>", block)
        rating = isnothing(rating_match) ? nothing : parse(Int, only(rating_match.captures))
        upvotes_match = match(r"data-reactions='.*?\"num\":(\d+)", block)
        upvotes = isnothing(upvotes_match) ? 0 : parse(Int, only(upvotes_match.captures))
        entry = Dict(
            "username" => username,
            "text" => final_text,
            "rating" => rating,
            "upvotes" => upvotes
        )
        push!(entries, entry)
    end
    (entries, occursin(r""">More Reviews</a>""", text))
end

function malweb_get_user(resource::Resource, username::String)
    url = "https://myanimelist.net/profile/$username"
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("malweb_get_user", r, url)
        return HTTP.Response(r)
    end
    if isempty(r.body)
        logerror("malweb_get_user received empty payload for $username with status $(r.status)")
        r = request(resource, "GET", url, Dict("impersonate" => true))
        if isempty(r.body)
            return HTTP.Response(500, [])
        end
    end

    asint(x) = parse(Int, replace(x, "," => ""))

    function info_panel(field)
        html_unescape(
            extract(
                r.body,
                """<span class="user-status-title di-ib fl-l fw-b">$field</span><span class="user-status-data di-ib fl-r">""",
                "</span>";
                optional = true,
            ),
        )
    end

    function info_links(field)
        asint(
            extract(
                r.body,
                """<span class="user-status-title di-ib fl-l fw-b">$field</span><span class="user-status-data di-ib fl-r fw-b">""",
                "</span>";
            )
        )
    end

    function item_counts(field)
        asint(
            extract(
                r.body,
                """/$field/.*?<span class="di-ib fl-l fn-grey2">Total Entries</span><span class="di-ib fl-r">""",
                "</span>",
            )
        )
    end

    function get_avatar()
        stem = "https://cdn.myanimelist.net/s/common/userimages/"
        leaf = extract(r.body, """<img class="lazyload" data-src="$stem""", "\"", optional=true)
        if isnothing(leaf)
            return nothing
        end
        stem * leaf
    end

    function get_about()
        matches = extract(r.body, """<div class="word-break">""", "</div>", optional=true, multiple = true)
        if isnothing(matches)
            return matches
        end
        html_unescape(first(matches))
    end

    ret = Dict(
        "version" => API_VERSION,
        "username" => username,
        "userid" => asint(
            extract(
                r.body,
                """href="https://myanimelist.net/modules.php\\?go=report&amp;type=profile&amp;id=""",
                "\"",
            )
        ),
        "last_online" => info_panel("Last Online"),
        "gender" => info_panel("Gender"),
        "birthday" => info_panel("Birthday"),
        "location" => info_panel("Location"),
        "joined" => info_panel("Joined"),
        "forum_posts" => info_links("Forum Posts"),
        "reviews" => info_links("Reviews"),
        "recommendations" => info_links("Recommendations"),
        "interest_stacks" => info_links("Interest Stacks"),
        "blog_posts" => info_links("Blog Posts"),
        "clubs" => info_links("Clubs"),
        "manga_count" => item_counts("mangalist"),
        "anime_count" => item_counts("animelist"),
        "avatar" => get_avatar(),
        "about" => get_about(),
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function malweb_get_image(resource::Resource, url::String)
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("malweb_get_image", r, url)
        return HTTP.Response(r)
    end
    ret = Dict("version" => API_VERSION, "url" => url, "image" => r.body)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/anilist" function anilist_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    if resource["location"] != "anilist"
        logerror("""anilist_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "list"
            return anilist_get_list(resource, data["userid"], data["medium"], data["chunk"])
        elseif endpoint == "fingerprint"
            return anilist_get_fingerprint(resource, data["userid"], data["medium"])
        elseif endpoint == "media"
            return anilist_get_media(resource, data["medium"], data["itemid"])
        elseif endpoint == "userid"
            return anilist_get_userid(resource, data["username"])
        elseif endpoint == "user"
            return anilist_get_user(resource, data["userid"])
        elseif endpoint == "image"
            return anilist_get_image(resource, data["url"])
        else
            logerror("anilist_api invalid endpoint $endpoint")
            return HTTP.Response(500, [])
        end
    catch e
        args = Dict(k => v for (k, v) in data if k != "token")
        logerror("anilist_api error $e for $args")
        return HTTP.Response(500, [])
    end
end

function anilist_get_list(resource::Resource, userid::Integer, medium::String, chunk::Integer)
    url = "https://graphql.anilist.co"
    query = """
    query (\$userID: Int, \$MEDIA: MediaType, \$chunk: Int, \$perChunk: Int) {
        MediaListCollection (userId: \$userID, type: \$MEDIA, chunk: \$chunk, perChunk: \$perChunk) {
            hasNextChunk
            lists {
                name
                isCustomList
                entries {
                    mediaId
                    status
                    score(format: POINT_10_DECIMAL)
                    progress
                    progressVolumes
                    repeat
                    priority
                    private
                    notes
                    advancedScores
                    startedAt {
                        year
                        month
                        day
                    }
                    completedAt {
                        year
                        month
                        day
                    }
                    updatedAt
                    createdAt
                }
            }
        }
    }
    """
    variables = Dict(
        "userID" => userid,
        "MEDIA" => uppercase(medium),
        "chunk" => chunk,
        "perChunk" => 500,
    )
    r = request(
        resource,
        "POST",
        url,
        encode(Dict("query" => query, "variables" => variables), :json)...,
    )
    if r.status >= 400
        logstatus("anilist_get_list", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    entries = Dict()
    for x in json["data"]["MediaListCollection"]["lists"]
        for entry in x["entries"]
            key = entry["mediaId"]
            if key ∉ keys(entries)
                entries[key] = Dict(
                    "version" => API_VERSION,
                    "userid" => userid,
                    "medium" => medium,
                    "itemid" => entry["mediaId"],
                    "status" => entry["status"],
                    "score" => entry["score"],
                    "progress" => entry["progress"],
                    "progressVolumes" => entry["progressVolumes"],
                    "repeat" => entry["repeat"],
                    "priority" => entry["priority"],
                    "private" => entry["private"],
                    "notes" => entry["notes"],
                    "listnames" => String[],
                    "advancedScores" => entry["advancedScores"],
                    "startedAt" => entry["startedAt"],
                    "completedAt" => entry["completedAt"],
                    "updatedAt" => entry["updatedAt"],
                    "createdAt" => entry["createdAt"],
                )
            end
            if x["isCustomList"]
                push!(entries[key]["listnames"], x["name"])
            end
        end
    end
    has_next_chunk = json["data"]["MediaListCollection"]["hasNextChunk"]
    ret = Dict(
        "entries" => collect(values(entries)),
        "chunk" => chunk,
        "next" => has_next_chunk,
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function anilist_get_fingerprint(resource::Resource, userid::Integer, medium::String)
    url = "https://graphql.anilist.co"
    query = """
    query (\$userID: Int, \$MEDIA: MediaType, \$chunk: Int, \$perChunk: Int, \$sort: [MediaListSort]) {
        MediaListCollection (userId: \$userID, type: \$MEDIA, chunk: \$chunk, perChunk: \$perChunk, sort: \$sort) {
            lists {
                entries {
                    mediaId
                    updatedAt
                }
            }
        }
    }
    """
    variables = Dict(
        "userID" => userid,
        "MEDIA" => uppercase(medium),
        "chunk" => 1,
        "perChunk" => 1,
        "sort" => "UPDATED_TIME_DESC",
    )
    r = request(
        resource,
        "POST",
        url,
        encode(Dict("query" => query, "variables" => variables), :json)...,
    )
    if r.status >= 400
        logstatus("anilist_get_fingerprint", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    entries = Dict()
    for x in json["data"]["MediaListCollection"]["lists"]
        for entry in x["entries"]
            key = entry["mediaId"]
            if key ∉ keys(entries)
                entries[key] = Dict(
                    "version" => API_VERSION,
                    "medium" => medium,
                    "itemid" => entry["mediaId"],
                    "updatedat" => entry["updatedAt"],
                )
            end
        end
    end
    ret = Dict(
        "entries" => collect(values(entries)),
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function anilist_get_userid(resource::Resource, username::String)
    url = "https://graphql.anilist.co"
    query = "query (\$username: String) { User (name: \$username) { id } }"
    variables = Dict("username" => username)
    r = request(
        resource,
        "POST",
        url,
        encode(Dict("query" => query, "variables" => variables), :json)...,
    )
    if r.status >= 400
        logerror("anilist_get_userid received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r)
    end
    ret = Dict(
        "version" => API_VERSION,
        "username" => username,
        "userid" => JSON3.read(r.body)["data"]["User"]["id"],
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function anilist_get_media(resource::Resource, medium::String, itemid::Integer)
    url = "https://graphql.anilist.co"
    initial_query = """
    query (\$id: Int, \$MEDIA: MediaType)
    {
        Media (id: \$id, type:\$MEDIA) {
            id
            idMal
            title {
                romaji
                english
                native
            }
            format
            status(version: 2)
            description
            startDate {
                year
                month
                day
            }
            endDate {
                year
                month
                day
            }
            season
            seasonYear
            episodes
            duration
            chapters
            volumes
            countryOfOrigin
            isLicensed
            source(version: 3)
            hashtag
            trailer {
                id
                site
                thumbnail
            }
            updatedAt
            coverImage {
                medium
                large
                extraLarge
            }
            bannerImage
            genres
            synonyms
            isLocked
            tags {
                name
                description
                category
                rank
                isGeneralSpoiler
                isMediaSpoiler
            }
            relations {
                edges {
                    node {
                        id
                        type
                    }
                    relationType(version: 2)
                }
            }
            characters(sort: RELEVANCE, perPage: 25, page: {characters_page}) {
                edges {
                    role
                    name
                    node {
                        name {
                            full
                            alternative
                            alternativeSpoiler
                        }
                        image {
                            large
                            medium
                        }
                        description
                        gender
                        dateOfBirth {
                            year
                            month
                            day
                        }
                        age
                        bloodType
                        favourites
                    }
                }
                pageInfo {
                    hasNextPage
                }
            }
            staff(sort: RELEVANCE) {
                edges {
                    role
                    node {
                        name {
                            full
                        }
                    }
                }
            }
            studios(sort: FAVOURITES_DESC) {
                nodes {
                    name
                }
            }
            isAdult
            externalLinks {
                url
            }
            reviews(sort: RATING_DESC, perPage: 25, page: {reviews_page}) {
                nodes {
                    userId
                    summary
                    body
                    rating
                    ratingAmount
                    score
                    createdAt
                    updatedAt
                }
                pageInfo {
                    hasNextPage
                }
            }
            recommendations(sort: RATING_DESC, perPage: 25, page: {recommendations_page}) {
                nodes {
                    rating
                    mediaRecommendation {
                        id
                        type
                    }
                }
                pageInfo {
                    hasNextPage
                }
            }
            isRecommendationBlocked
            isReviewBlocked
            modNotes
        }
    }"""
    function get_paged_query(components)
        header = """
    query (\$id: Int, \$MEDIA: MediaType)
    {
        Media (id: \$id, type:\$MEDIA) {
        """
        footer = """
        }
    }"""
        paged_components = Dict(
            "reviews" => """
            reviews(sort: RATING_DESC, perPage: 25, page: {reviews_page}) {
                nodes {
                    userId
                    summary
                    body
                    rating
                    ratingAmount
                    score
                    createdAt
                    updatedAt
                }
                pageInfo {
                    hasNextPage
                }
            }""",
            "recommendations" => """
            recommendations(sort: RATING_DESC, perPage: 25, page: {recommendations_page}) {
                nodes {
                    rating
                    mediaRecommendation {
                        id
                        type
                    }
                }
                pageInfo {
                    hasNextPage
                }
            }
            """,
            "characters" => """
            characters(sort: RELEVANCE, perPage: 25, page: {characters_page}) {
                edges {
                    role
                    name
                    node {
                        name {
                            full
                            alternative
                            alternativeSpoiler
                        }
                        image {
                            large
                            medium
                        }
                        description
                        gender
                        dateOfBirth {
                            year
                            month
                            day
                        }
                        age
                        bloodType
                        favourites
                    }
                }
                pageInfo {
                    hasNextPage
                }
            }
            """
        )
        query = header
        for k in components
            query *= paged_components[k]
        end
        query *= footer
        query
    end
    function season(year, season)
        if isnothing(year) || isnothing(season)
            return nothing
        end
        "$year $season"
    end
    details = Dict()
    relations = []
    pages = Dict("reviews" => 1, "recommendations" => 1, "characters" => 1)
    while !isempty(pages)
        variables = Dict("id" => itemid, "MEDIA" => uppercase(medium))
        is_first_page = all(values(pages) .== 1)
        if is_first_page
            query = initial_query
        else
            query = get_paged_query(collect(keys(pages)))
        end
        for (k, v) in pages
            query = replace(query, "{$(k)_page}" => v)
        end
        r = request(
            resource,
            "POST",
            url,
            encode(Dict("query" => query, "variables" => variables), :json)...,
        )
        if r.status >= 400
            logstatus("anilist_get_media", r, url)
            return HTTP.Response(r)
        end
        json = JSON3.read(r.body)
        data = json["data"]["Media"]
        if is_first_page
            page_details = Dict(
                "version" => API_VERSION,
                "itemid" => data["id"],
                "malid" => data["idMal"],
                "medium" => medium,
                "title" => data["title"],
                "mediatype" => optget(data, "format"),
                "status" => optget(data, "status"),
                "summary" => optget(data, "description"),
                "startdate" => optget(data, "startDate"),
                "enddate" => optget(data, "endDate"),
                "seasonYear" => optget(data, "seasonYear"),
                "season" => optget(data, "season"),
                "episodes" => optget(data, "episodes"),
                "duration" => optget(data, "duration"),
                "chapters" => optget(data, "chapters"),
                "volumes" => optget(data, "volumes"),
                "countryOfOrigin" => optget(data, "countryOfOrigin"),
                "isLicensed" => optget(data, "isLicensed"),
                "source" => optget(data, "source"),
                "hashtag" => optget(data, "hashtag"),
                "trailer" => optget(data, "trailer"),
                "updatedAt" => optget(data, "updatedAt"),
                "coverImage" => optget(data, "coverImage"),
                "bannerimage" => optget(data, "bannerImage"),
                "genres" => optget(data, "genres"),
                "synonyms" => optget(data, "synonyms"),
                "isLocked" => optget(data, "isLocked"),
                "tags" => optget(data, "tags"),
                "charactersPeek" => [
                    Dict(
                        "role" => x["role"],
                        "name" => x["name"],
                        "node" => x["node"],
                    )
                    for x in get(data["characters"], "edges", [])
                ],
                "staffPeek" => [
                    Dict("role" => x["role"], "name" => x["node"]["name"]["full"])
                    for x in get(data["staff"], "edges", [])
                ],
                "studios" => [x["name"] for x in get(data["studios"], "nodes", [])],
                "externalUrls" =>
                    [x["url"] for x in get(data["studios"], "externalLinks", [])],
                "isAdult" => optget(data, "isAdult"),
                "reviewsPeek" => collect(get(data["reviews"], "nodes", [])),
                "recommendationsPeek" => [
                    Dict(
                        "rating" => x["rating"],
                        "medium" => x["mediaRecommendation"]["type"],
                        "itemid" => x["mediaRecommendation"]["id"],
                    ) for x in get(data["recommendations"], "nodes", []) if
                    !isnothing(x["mediaRecommendation"])
                ],
                "isRecommendationBlocked" => optget(data, "isRecommendationBlocked"),
                "isReviewBlocked" => optget(data, "isReviewBlocked"),
                "modNotes" => optget(data, "modNotes"),
            )
            page_relations = []
            for e in get(data["relations"], "edges", [])
                d = Dict(
                    "version" => API_VERSION,
                    "relation" => e["relationType"],
                    "itemid" => itemid,
                    "medium" => medium,
                    "target_id" => e["node"]["id"],
                    "target_medium" => e["node"]["type"],
                )
                push!(page_relations, d)
            end
        else
            page_details = Dict()
            if "reviews" in keys(pages)
                page_details["reviewsPeek"] = collect(get(data["reviews"], "nodes", []))
            end
            if "recommendations" in keys(pages)
                page_details["recommendationsPeek"] = [
                    Dict(
                        "rating" => x["rating"],
                        "medium" => x["mediaRecommendation"]["type"],
                        "itemid" => x["mediaRecommendation"]["id"],
                    ) for x in get(data["recommendations"], "nodes", []) if
                    !isnothing(x["mediaRecommendation"])
                ]
            end
            if "characters" in keys(pages)
                page_details["charactersPeek"] = [
                    Dict(
                        "role" => x["role"],
                        "name" => x["name"],
                        "node" => x["node"],
                    )
                    for x in get(data["characters"], "edges", [])
                ]
            end
            page_relations = []
        end
        for k in collect(keys(pages))
            if data[k]["pageInfo"]["hasNextPage"]
                pages[k] += 1
            else
                delete!(pages, k)
            end
        end
        for (k, v) in page_details
            if k ∉ keys(details)
                details[k] = v
            else
                append!(details[k], v)
            end
        end
        append!(relations, page_relations)
    end
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function anilist_get_user(resource::Resource, userid::Integer)
    url = "https://graphql.anilist.co"
    query = """
    query (\$id: Int)
    {
        User (id: \$id) {
            id
            name
            about
            avatar {
                medium
                large
            }
            bannerImage
            options {
                titleLanguage
                displayAdultContent
            }
            favourites {
                manga {
                    edges {
                        favouriteOrder
                        node {
                            id
                            type
                        }
                    }
                }
                anime {
                    edges {
                        favouriteOrder
                        node {
                            id
                            type
                        }
                    }
                }
                characters {
                    edges {
                        favouriteOrder
                        node {
                            name {
                                full
                            }
                        }
                    }
                }
                staff {
                    edges {
                        favouriteOrder
                        node {
                            name {
                                full
                            }
                        }
                    }
                }
                studios {
                    edges {
                        favouriteOrder
                        node {
                            name
                        }
                    }
                }
            }
            statistics {
                manga {
                    count
                }
                anime {
                    count
                }
            }
            unreadNotificationCount
            donatorTier
            donatorBadge
            moderatorRoles
            createdAt
            updatedAt
            previousNames {
                name
                createdAt
                updatedAt
            }
        }
    }"""
    variables = Dict("id" => userid)
    r = request(
        resource,
        "POST",
        url,
        encode(Dict("query" => query, "variables" => variables), :json)...,
    )
    if r.status >= 400
        logstatus("anilist_get_user", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    data = json["data"]["User"]
    ret = Dict(
        "version" => API_VERSION,
        "userid" => data["id"],
        "username" => data["name"],
        "about" => data["about"],
        "avatar" => data["avatar"],
        "bannerImage" => data["bannerImage"],
        "titleLanguage" => data["options"]["titleLanguage"],
        "displayAdultContent" => data["options"]["displayAdultContent"],
        "mangaFavorites" => [
            Dict(
                "order" => x["favouriteOrder"],
                "itemid" => x["node"]["id"],
                "type" => x["node"]["type"],
            ) for x in data["favourites"]["manga"]["edges"]
        ],
        "animeFavorites" => [
            Dict(
                "order" => x["favouriteOrder"],
                "itemid" => x["node"]["id"],
                "type" => x["node"]["type"],
            ) for x in data["favourites"]["anime"]["edges"]
        ],
        "characterFavorites" => [
            Dict("order" => x["favouriteOrder"], "name" => x["node"]["name"]["full"]) for x in data["favourites"]["characters"]["edges"]
        ],
        "staffFavorites" => [
            Dict("order" => x["favouriteOrder"], "name" => x["node"]["name"]["full"]) for x in data["favourites"]["staff"]["edges"]
        ],
        "studioFavorites" => [
            Dict("order" => x["favouriteOrder"], "name" => x["node"]["name"]) for
            x in data["favourites"]["studios"]["edges"]
        ],
        "mangaCount" => data["statistics"]["manga"]["count"],
        "animeCount" => data["statistics"]["anime"]["count"],
        "unreadNotificationCount" => data["unreadNotificationCount"],
        "donatorTier" => data["donatorTier"],
        "donatorBadge" => data["donatorBadge"],
        "moderatorRoles" => data["moderatorRoles"],
        "createdAt" => data["createdAt"],
        "updatedAt" => data["updatedAt"],
        "previousNames" => data["previousNames"],
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function anilist_get_image(resource::Resource, url::String)
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("anilist_get_image", r, url)
        return HTTP.Response(r)
    end
    ret = Dict("version" => API_VERSION, "url" => url, "image" => r.body)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/kitsu" function kitsu_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    if resource["location"] != "kitsu"
        logerror("""kitsu_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "list"
            return kitsu_get_list(resource, data["auth"], data["userid"], data["offset"])
        elseif endpoint == "fingerprint"
            return kitsu_get_fingerprint(resource, data["auth"], data["userid"])
        elseif endpoint == "media"
            return kitsu_get_media(resource, data["auth"], data["medium"], data["itemid"])
        elseif endpoint == "userid"
            return kitsu_get_userid(resource, data["auth"], data["username"], data["key"])
        elseif endpoint == "user"
            return kitsu_get_user(resource, data["auth"], data["userid"])
        elseif endpoint == "image"
            return kitsu_get_image(resource, data["url"])
        elseif endpoint == "token"
            return kitsu_get_token(resource)
        else
            logerror("kitsu_api invalid endpoint $endpoint")
            return HTTP.Response(500, [])
        end
    catch e
        args = Dict(k => v for (k, v) in data if k != "token")
        logerror("kitsu_api error $e for $args")
        return HTTP.Response(500, [])
    end
end

function kitsu_get_token(resource::Resource)
    credentials = resource["credentials"]
    username = credentials["username"]
    password = credentials["password"]
    url = "https://kitsu.app/api/oauth/token"
    body = Dict("grant_type" => "password", "username" => username, "password" => password)
    headers, content = encode(body, :json)
    headers = Dict{String,Any}(headers)
    headers["impersonate"] = true
    r = request(resource, "POST", url, headers, content)
    if r.status >= 400
        logerror("kitsu_get_token failed for $username")
        return HTTP.Response(r)
    end
    data = JSON3.read(r.body)
    token = data["access_token"]
    expires_in = data["expires_in"]
    expiry_time = time() + expires_in
    ret = Dict("token" => token, "expiry_time" => expiry_time)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function kitsu_get_userid(resource::Resource, auth::String, username::String, key::String)
    url = string(
        HTTP.URI(
            "https://kitsu.app/api/edge/users";
            query = Dict("filter[$key]" => username),
        ),
    )
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => true)
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("kitsu_get_userid", r, url)
        return HTTP.Response(r)
    end
    data = JSON3.read(r.body)["data"]
    if length(data) != 1
        return HTTP.Response(404, [])
    end
    ret = Dict("version" => API_VERSION, "userid" => parse(Int, only(data)["id"]))
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function kitsu_get_media(resource::Resource, auth::String, medium::String, itemid::Integer)
    params = Dict(
        "include" => "genres,mappings,mediaRelationships.destination",
        "fields[genres]" => "name",
        "fields[mappings]" => "externalSite,externalId",
    )
    url = string(HTTP.URI("https://kitsu.app/api/edge/$medium/$itemid"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => true)
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("kitsu_get_media", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    data = json["data"]["attributes"]
    function extractid(x)
        try
            if isempty(x)
                return nothing
            end
            return parse(Int, first(x))
        catch
            logerror("kitsu_get_media could not parse Int $x")
            return nothing
        end
    end
    details = Dict(
        "version" => API_VERSION,
        "medium" => medium,
        "itemid" => itemid,
        "createdAt" => data["createdAt"],
        "updatedAt" => data["updatedAt"],
        "synopsis" => data["synopsis"],
        "titles" => data["titles"],
        "canonicalTitle" => data["canonicalTitle"],
        "startDate" => data["startDate"],
        "endDate" => data["endDate"],
        "ageRating" => data["ageRating"],
        "ageRatingGuide" => data["ageRatingGuide"],
        "subtype" => data["subtype"],
        "status" => data["status"],
        "posterImage" => data["posterImage"],
        "coverImage" => data["coverImage"],
        "episodeCount" => optget(data, "episodeCount"),
        "episodeLength" => optget(data, "episodeLength"),
        "chapterCount" => optget(data, "chapterCount"),
        "volumeCount" => optget(data, "volumeCount"),
        "youtubeVideoId" => optget(data, "youtubeVideoId"),
        "nsfw" => optget(data, "nsfw"),
        "genres" => nothing,
        "malid" => nothing,
        "anilistid" => nothing,
    )
    relations = []
    if "included" in keys(json)
        details["genres"] = [x["attributes"]["name"] for x in json["included"] if x["type"] == "genres"]
        details["malid"] = extractid([
            x["attributes"]["externalId"] for
            x in json["included"] if x["type"] == "mappings" &&
            startswith(x["attributes"]["externalSite"], "myanimelist")
        ])
        details["anilistid"] = extractid([
            x["attributes"]["externalId"] for
            x in json["included"] if x["type"] == "mappings" &&
            startswith(x["attributes"]["externalSite"], "anilist")
        ])
        for x in filter(r -> r["type"] == "mediaRelationships", json["included"])
            d = Dict(
                "version" => API_VERSION,
                "relation" => x["attributes"]["role"],
                "itemid" => itemid,
                "medium" => medium,
                "target_id" => parse(Int, x["relationships"]["destination"]["data"]["id"]),
                "target_medium" => x["relationships"]["destination"]["data"]["type"],
            )
            push!(relations, d)
        end
    end
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function kitsu_get_list(
    resource::Resource,
    auth::String,
    userid::Integer,
    offset::Integer,
)
    url = "https://kitsu.io/api/edge/library-entries"
    params = Dict(
        "filter[user_id]" => userid,
        "include" => "media",
        "fields[media]" => "id",
        "page[limit]" => 500,
    )
    if offset > 0
        params["page[offset]"] = offset
    end
    url = string(HTTP.URI("https://kitsu.app/api/edge/library-entries"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => true)
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("kitsu_get_list", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    kitsu_rating(T, x::Nothing) = nothing
    kitsu_rating(T, x::String) = parse(T, x)
    kitsu_rating(T, x::Real) = convert(T, x)
    entries = []
    for x in json["data"]
        d = Dict(
            "version" => API_VERSION,
            "userid" => userid,
            "medium" => x["relationships"]["media"]["data"]["type"],
            "itemid" => parse(Int, x["relationships"]["media"]["data"]["id"]),
            "status" => x["attributes"]["status"],
            "progress" => x["attributes"]["progress"],
            "volumesOwned" => optget(x["attributes"], "volumesOwned"),
            "reconsuming" => x["attributes"]["reconsuming"],
            "reconsumeCount" => x["attributes"]["reconsumeCount"],
            "notes" => x["attributes"]["notes"],
            "private" => x["attributes"]["private"],
            "updatedAt" => optget(x["attributes"], "updatedAt"),
            "createdAt" => optget(x["attributes"], "createdAt"),
            "startedAt" => optget(x["attributes"], "startedAt"),
            "finishedAt" => optget(x["attributes"], "finishedAt"),
            "progressedAt" => optget(x["attributes"], "progressedAt"),
            "reactionSkipped" => x["attributes"]["reactionSkipped"],
            "ratingTwenty" => kitsu_rating(Int, optget(x["attributes"], "ratingTwenty")),
            "rating" => kitsu_rating(Float64, optget(x["attributes"], "rating")),
        )
        push!(entries, d)
    end
    ret = Dict("entries" => entries, "offset" => offset)
    if "next" in keys(json["links"])
        ret["next"] = json["links"]["next"]
    end
    ret["limit"] = params["page[limit]"]
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function kitsu_get_fingerprint(
    resource::Resource,
    auth::String,
    userid::Integer,
)
    url = "https://kitsu.io/api/edge/library-entries"
    params = Dict(
        "filter[user_id]" => userid,
        "page[limit]" => 1,
        "sort" => "-updatedAt",
        "fields[library-entries]" => "updatedAt",
    )
    url = string(HTTP.URI("https://kitsu.app/api/edge/library-entries"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => true)
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("kitsu_get_fingerprint", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    entries = []
    for x in json["data"]
        d = Dict(
            "version" => API_VERSION,
            "updatedat" => optget(x["attributes"], "updatedAt"),
        )
        push!(entries, d)
    end
    ret = Dict("entries" => entries)
    HTTP.Response(200, encode(ret, :msgpack)...)
end


function kitsu_get_user(resource::Resource, auth::String, userid::Integer)
    params = Dict("include" => "stats")
    url = string(HTTP.URI("https://kitsu.app/api/edge/users/$userid"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => true)
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logstatus("kitsu_get_user", r, url)
        return HTTP.Response(r)
    end
    json = JSON3.read(r.body)
    data = json["data"]["attributes"]
    ret = Dict(
        "version" => API_VERSION,
        "userid" => userid,
        "createdAt" => data["createdAt"],
        "updatedAt" => data["updatedAt"],
        "name" => data["name"],
        "pastNames" => data["pastNames"],
        "slug" => data["slug"],
        "about" => data["about"],
        "location" => data["location"],
        "waifuOrHusbando" => data["waifuOrHusbando"],
        "followersCount" => data["followersCount"],
        "followingCount" => data["followingCount"],
        "birthday" => data["birthday"],
        "gender" => data["gender"],
        "commentsCount" => data["commentsCount"],
        "favoritesCount" => data["favoritesCount"],
        "likesGivenCount" => data["likesGivenCount"],
        "reviewsCount" => data["reviewsCount"],
        "likesReceivedCount" => data["likesReceivedCount"],
        "ratingsCount" => data["ratingsCount"],
        "mediaReactionsCount" => data["mediaReactionsCount"],
        "title" => data["title"],
        "profileCompleted" => data["profileCompleted"],
        "feedCompleted" => data["feedCompleted"],
        "proTier" => data["proTier"],
        "avatar" => data["avatar"],
        "coverImage" => data["coverImage"],
        "status" => data["status"],
        "subscribedToNewsletter" => data["subscribedToNewsletter"],
    )
    if "included" in keys(json)
        included = json["included"]
        maybe_unpack(x) = length(x) == 0 ? 0 : only(x)
        ret["manga_count"] = maybe_unpack([
            x["attributes"]["statsData"]["media"] for x in included if x["attributes"]["kind"] == "manga-amount-consumed"
        ])
        ret["anime_count"] = maybe_unpack([
            x["attributes"]["statsData"]["media"] for x in included if x["attributes"]["kind"] == "anime-amount-consumed"
        ])
    else
        ret["manga_count"] = 0
        ret["anime_count"] = 0
    end
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function kitsu_get_image(resource::Resource, url::String)
    r = request(resource, "GET", url, Dict("impersonate" => true))
    if r.status >= 400
        logstatus("kitsu_get_image", r, url)
        return HTTP.Response(r)
    end
    ret = Dict("version" => API_VERSION, "url" => url, "image" => r.body)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

Oxygen.@post "/animeplanet" function animeplanet_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    sessionid = data["sessionid"]
    if resource["location"] != "animeplanet"
        logerror("""animeplanet_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "list"
            return animeplanet_get_list(
                resource,
                sessionid,
                data["medium"],
                data["username"],
                data["page"],
                data["expand_pagelimit"],
            )
        elseif endpoint == "fingerprint"
            return animeplanet_get_fingerprint(
                resource,
                sessionid,
                data["medium"],
                data["username"],
            )
        elseif endpoint == "media"
            return animeplanet_get_media(
                resource,
                sessionid,
                data["medium"],
                data["itemid"],
            )
        elseif endpoint == "user"
            return animeplanet_get_user(resource, sessionid, data["username"])
        elseif endpoint == "username"
            return animeplanet_get_username(resource, sessionid, data["userid"])
        elseif endpoint == "feed"
            return animeplanet_get_feed(
                resource,
                sessionid,
                data["medium"],
                data["username"],
            )
        elseif endpoint == "image"
            return animeplanet_get_image(resource, sessionid, data["url"])
        elseif endpoint == "login"
            return animeplanet_login(resource, sessionid)
        else
            logerror("animeplanet_api invalid endpoint $endpoint")
            return HTTP.Response(500, [])
        end
    catch e
        args = Dict(k => v for (k, v) in data if k != "token")
        logerror("animeplanet_api error $e for $args")
        return HTTP.Response(500, [])
    end
end

function parse_animeplanet_response(resource::Resource, r::Response, found::Function)
    text = r.body
    if occursin("<title>Just a moment...</title>", text)
        return HTTP.Response(401, [])
    end
    if r.status >= 400
        return HTTP.Response(r.status, [])
    end
    if !found(text)
        return HTTP.Response(404, [])
    end
    text
end

function animeplanet_get_list(
    resource::Resource,
    sessionid::String,
    medium::String,
    username::String,
    page::Integer,
    expand_pagelimit::Bool,
)
    if expand_pagelimit
        params = Dict("per_page" => 560)
        url = string(
            HTTP.URI("https://www.anime-planet.com/users/$username/$medium"; query = params),
        )
        r = request(resource, "GET", url, Dict("sessionid" => sessionid))
        text = parse_animeplanet_response(resource, r, x -> !occursin("<title>Search Results for $username", x))
        if text isa HTTP.Response
            logstatus("animeplanet_get_list", text, url)
            return text
        end
        ret = Dict("nextpage" => page)
        return HTTP.Response(200, encode(ret, :msgpack)...)
    end
    params = Dict{String,Any}("sort" => "user_updated", "order" => "desc")
    if page != 1
        params["page"] = page
    end
    url = string(
        HTTP.URI("https://www.anime-planet.com/users/$username/$medium"; query = params),
    )
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    text = parse_animeplanet_response(resource, r, x -> !occursin("<title>Search Results for $username", x))
    if text isa HTTP.Response
        logstatus("animeplanet_get_list", text, url)
        return text
    end
    if occursin("has chosen to make their content private.", text)
        return HTTP.Response(403, [])
    end
    default_pagelimit =
        occursin("""<option value="35" selected="selected">35</option>""", text)
    page_numbers = Set([
        parse(Int, only(m.captures)) for m in eachmatch(r"""&amp;page=([0-9]*)'>""", text)
    ])
    next_page = page + 1 in page_numbers
    if default_pagelimit && next_page
        ret = Dict("expand_pagelimit" => true, "nextpage" => page)
        return HTTP.Response(200, encode(ret, :msgpack)...)
    end
    function get_score(line)
        matches = extract(line, """<div class='ttRating'>""", "</div>", multiple = true)
        if isempty(matches)
            return nothing
        end
        parse(Float64, only(matches))
    end
    function get_progress(line)
        if medium == "manga"
            m = extract(line, """</span> """, " chs<", optional = true)
        elseif medium == "anime"
            m = extract(line, """</span> """, " eps<", optional = true)
        else
            @assert false
        end
        if isnothing(m)
            return nothing
        end
        parse(Int, m)
    end
    item_order = 0
    if page != 1
        item_order += 560 * (page - 1)
    end
    entries = []
    prevline = nothing
    for line in split(text, "\n")
        if occursin("<h3 class='cardName' >", line)
            d = Dict(
                "version" => API_VERSION,
                "username" => username,
                "medium" => medium,
                "itemid" => extract(prevline, """href="/$medium/""", '"'),
                "title" =>
                    html_unescape(extract(line, """<h3 class='cardName' >""", "</h3>")),
                "score" => get_score(line),
                "status" => parse(Int, extract(line, """<span class='status""", "'>")),
                "progress" => get_progress(line),
                "item_order" => item_order,
            )
            item_order += 1
            push!(entries, d)
        end
        prevline = line
    end
    ret = Dict{String, Any}("entries" => entries)
    if next_page
        ret["nextpage"] = page + 1
    end
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_fingerprint(
    resource::Resource,
    sessionid::String,
    medium::String,
    username::String,
)
    params = Dict{String,Any}("sort" => "user_updated", "order" => "desc")
    url = string(
        HTTP.URI("https://www.anime-planet.com/users/$username/$medium"; query = params),
    )
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    text = parse_animeplanet_response(resource, r, x -> !occursin("<title>Search Results for $username", x))
    if text isa HTTP.Response
        logstatus("animeplanet_get_fingerprint", text, url)
        return text
    end
    if occursin("has chosen to make their content private.", text)
        return HTTP.Response(403, [])
    end
    entries = []
    prevline = nothing
    for line in split(text, "\n")
        if occursin("<h3 class='cardName' >", line)
            d = Dict(
                "version" => API_VERSION,
                "medium" => medium,
                "itemid" => extract(prevline, """href="/$medium/""", '"'),
                "item_order" => 0,
            )
            push!(entries, d)
            break
        end
        prevline = line
    end
    ret = Dict("entries" => entries)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_media(
    resource::Resource,
    sessionid::String,
    medium::String,
    itemid::String,
)
    url = "https://www.anime-planet.com/$medium/$itemid"
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    text = parse_animeplanet_response(resource, r, x -> !occursin("<h1>You searched for $itemid", x))
    if text isa HTTP.Response
        logstatus("animeplanet_get_media", text, url)
        return text
    end

    function get_media_season(text)
        matches = extract(text, "/seasons/", "\">", optional = true, multiple = true)
        if length(matches) == 2 # first match is a link to the current season
            return matches[end]
        end
    end

    function get_media_studios(text)
        if medium == "manga"
            return vcat(
                extract(
                    text,
                    """<a href="/manga/magazines/.*?>""",
                    "</a>",
                    multiple = true,
                ),
                extract(
                    text,
                    """<a href="/manga/publishers/.*?>""",
                    "</a>",
                    multiple = true,
                ),
            )
        elseif medium == "anime"
            return extract(
                text,
                """<a href="/anime/studios/.*?>""",
                "</a>",
                multiple = true,
            )
        else
            @assert false
        end
    end

    function get_type(text)
        if medium == "manga"
            return extract(
                text,
                """<section class="pure-g entryBar">(?s).*?<div class="pure-1 md-1-5">""",
                "</div>",
                optional = true,
            )
        elseif medium == "anime"
            extract(text, """<span class="type">""", "</span>", optional = true)
        else
            @assert false
        end
    end

    details = Dict(
        "version" => API_VERSION,
        "medium" => medium,
        "itemid" => itemid,
        "title" => html_unescape(
            extract(text, """<h1 itemprop="name".*?>""", "</h1>", optional = true),
        ),
        "alttitle" => html_unescape(
            extract(
                text,
                """<h2 class="aka">(?s).*?Alt title:""",
                "</h2>",
                optional = true,
            ),
        ),
        "year" =>
            extract(text, """<span class='iconYear'>""", "</span>", optional = true),
        "type" => get_type(text),
        "season" => get_media_season(text),
        "studios" => get_media_studios(text),
        "genres" => extract(text, """<a href="/$medium/tags/""", "\"", multiple = true),
        "summary" => html_unescape(
            extract(
                text,
                """property='og:description' content='""",
                "' />",
                optional = true,
            ),
        ),
        "image" => extract(
            text,
            """class="screenshots"(?s).*?src=\"""",
            "\"",
            optional = true,
        ),
    )
    relations = [
        d = Dict(
            "version" => API_VERSION,
            "relation" => "relation",
            "itemid" => itemid,
            "medium" => medium,
            "target_id" => m.captures[2],
            "target_medium" => m.captures[1],
        ) for m in eachmatch(
            Regex("""<a href="/(manga|anime)/(.*?)\" class="RelatedEntry"""),
            text,
        )
    ]
    recommendations = []
    recommendations_page = 1
    has_next_page = true
    while has_next_page
        url = "https://www.anime-planet.com/$medium/$itemid/recommendations/$medium?page=$recommendations_page"
        r = request(resource, "GET", url, Dict("sessionid" => sessionid))
        text = parse_animeplanet_response(resource, r, x -> !occursin("<h1>You searched for $itemid</h1>", x))
        if text isa HTTP.Response
            logstatus("animeplanet_get_media", text, url)
            return text
        end
        entries, has_next_page = animeplanet_get_media_recommendations(text, medium, itemid, recommendations_page)
        append!(recommendations, entries)
        recommendations_page += 1
    end
    details["recommendations"] = recommendations
    reviews = []
    reviews_page = 1
    has_next_page = true
    while has_next_page
        url = "https://www.anime-planet.com/$medium/$itemid/reviews?page=$reviews_page"
        r = request(resource, "GET", url, Dict("sessionid" => sessionid))
        text = parse_animeplanet_response(resource, r, x -> !occursin("<h1>You searched for $itemid</h1>", x))
        if text isa HTTP.Response
            logstatus("animeplanet_get_media", text, url)
            return text
        end
        entries, has_next_page = animeplanet_get_media_reviews(text, medium, itemid, reviews_page)
        append!(reviews, entries)
        reviews_page += 1
    end
    details["reviews"] = reviews
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_media_recommendations(text::String, medium::String, itemid::String, page::Int)
    entries = []
    item_chunks = split(text, r"""<h3 class="flipMain">""")
    user_rec_regex =
        r"(?s)<h5>.*?<a href=\"/users/([^\"]+)\">\1</a>.*?</h5>\s*<p class=\"recReason\">(.*?)</p>"
    for chunk in item_chunks[2:end]
        itemid_match = match(r"<a href=\"/(?:anime|manga)/([^\"]+)\">", chunk)
        if isnothing(itemid_match)
            continue
        end
        rec_itemid = only(itemid_match.captures)
        for user_match in eachmatch(user_rec_regex, chunk)
            username, raw_text = user_match.captures
            final_text =
                replace(raw_text, r"</p>\s*<p>" => "\n\n") |>
                html_unescape |>
                (s -> replace(s, r"<[^>]*>" => "")) |>
                strip
            push!(entries, Dict("itemid" => rec_itemid, "username" => username, "text" => final_text))
        end
    end
    has_next_page = occursin(Regex("<li class='next'><a href=?page=$page"), text)
    (entries, has_next_page)
end

function animeplanet_get_media_reviews(text::String, medium::String, itemid::String, page::Int)
    entries = []
    review_chunks = split(
        text,
        r"""<section class="pure-g" id="\d+" itemprop="review" itemscope itemtype="https://schema.org/Review">""",
    )
    if length(review_chunks) < 2
        return ([], false)
    end
    maybeint(x) = isnothing(x) ? 0 : parse(Int, x)
    for chunk in review_chunks[2:end]
        username = extract(chunk, """<span itemprop="name">""", "</span>", optional = true)
        if isnothing(username)
            continue
        end
        raw_text = extract(
            chunk,
            """<div class="pure-1 userContent readMore" itemprop="reviewBody">""",
            "</div>",
            optional = true,
        )
        if isnothing(raw_text)
            continue
        end
        final_text =
            replace(
                raw_text,
                r"<p>" => "",
                r"</p>" => "\n\n",
                r"<br\s*/?>" => "\n",
                r"<em>" => "",
                r"</em>" => "",
                r"<strong>" => "",
                r"</strong>" => "",
            ) |>
            html_unescape |>
            (s -> replace(s, r"<[^>]*>" => "")) |>
            strip

        rating_str = extract(
            chunk,
            """<meta itemprop="ratingValue" content=\"""",
            "\"",
            optional = true,
        )
        rating = isnothing(rating_str) ? nothing : round(Int, parse(Float64, rating_str))
        loves_str = extract(
            chunk,
            """>toggle_love\\('user_reviews', \\d+\\)">""",
            "</a>",
            optional = true,
        )
        funny_str = extract(chunk, """Funny  \\(""", "\\)", optional = true)
        helpful_str = extract(chunk, """Helpful  \\(""", "\\)", optional = true)
        upvotes = maybeint(loves_str) + maybeint(funny_str) + maybeint(helpful_str)
        entry = Dict(
            "username" => username,
            "text" => final_text,
            "rating" => rating,
            "upvotes" => upvotes,
        )
        push!(entries, entry)
    end
    has_next_page = occursin(Regex("<li class='next'><a href=?page=$page"), text)
    (entries, has_next_page)
end

function animeplanet_get_user(resource::Resource, sessionid::String, username::String)
    url = "https://www.anime-planet.com/users/$username"
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    function found(text::AbstractString)
        try
            page_username = extract(text, """<meta name="description" content="Meet """, " on Anime-Planet.")
            if lowercase(page_username) != lowercase(username)
                logerror("animeplanet_get_user mismatched usernames $username $page_username")
                @assert false
            end
        catch
            return false
        end
        true
    end
    text = parse_animeplanet_response(resource, r, found)
    if text isa HTTP.Response
        logstatus("animeplanet_get_user", r, url)
        return text
    end
    anime_counts = extract(
        text,
        "the ",
        " anime they&#039;ve watched",
        capture = "([0-9,]+)",
        multiple = true,
    )
    manga_counts = extract(
        text,
        "the ",
        " manga they&#039;ve read",
        capture = "([0-9,]+)",
        multiple = true,
    )
    maybeint(x, default) = isempty(x) ? default : parse(Int, x)
    function parse_item_count(x::Vector)
        if isempty(x)
            return 0
        end
        maybeint(replace(first(x), "," => ""), 0)
    end
    function to_image_url(x)
        if isnothing(x) || !startswith(x, "/")
            return nothing
        end
        x = first(split(x, "?"))
        "https://www.anime-planet.com$x"
    end
    ret = Dict(
        "version" => API_VERSION,
        "username" => username,
        "userid" => maybeint(extract(text, """<a href="/forum/members/.*\\.""", "\">forum</a>"), nothing),
        "about" => html_unescape(
            extract(text, """<section class="profBio userContent">""", "</section>"),
        ),
        "joined" =>
            extract(text, """<i class="fa fa-calendar"></i>""", "<", optional = true),
        "age" => extract(text, """<i class="fa fa-user"></i>""", "<", optional = true),
        "location" => html_unescape(
            extract(text, """<i class="fa fa-home"></i>""", "<", optional = true),
        ),
        "anime_count" => parse_item_count(anime_counts),
        "manga_count" => parse_item_count(manga_counts),
        "last_online" => extract(
            text,
            """<h1 id="profileName" class="tooltip" title=\"""",
            "\"",
            optional = true,
        ),
        "followers" => parse(Int, extract(text, """followers">""", " Followers<")),
        "following" => parse(Int, extract(text, """following">""", " Following<")),
        "avatar" => to_image_url(extract(text, "<meta property='og:image' content='", "'", optional = true)),
        "banner_image" => to_image_url(extract(text, "style='background-image: url\\(", "'", optional = true)),
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_feed(
    resource::Resource,
    sessionid::String,
    medium::String,
    username::String,
)
    params = Dict("type" => medium)
    url = string(
        HTTP.URI("https://www.anime-planet.com/users/$username/feed", query = params),
    )
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    text = parse_animeplanet_response(resource, r, x -> !occursin("<h1>You searched for $username</h1>", x))
    if text isa HTTP.Response
        logstatus("animeplanet_get_feed", text, url)
        return text
    end
    feed_entries = [x for x in split(text, "\n") if occursin("data-timestamp", x)]
    ret = Dict{String,Int}()
    for x in reverse(feed_entries)
        title = extract(x, """href="/$medium/""", "\"")
        updated_at = parse(Int, extract(x, "data-timestamp=\"", "\">"))
        ret[title] = updated_at
    end
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_username(resource::Resource, sessionid::String, userid::Integer)
    url = "https://www.anime-planet.com/forum/members/$userid"
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    text = parse_animeplanet_response(resource, r, x -> !occursin("The requested user could not be found.", x))
    if text isa HTTP.Response
        logstatus("animeplanet_get_username", text, url)
        return text
    end
    username = extract(
        text,
        "<title>",
        " \\| Anime-Planet Forum</title>",
        capture = "([a-zA-Z0-9_]+?)",
        optional=true,
    )
    if isnothing(username)
        return HTTP.Response(404, [])
    end
    ret = Dict(
        "version" => API_VERSION,
        "userid" => userid,
        "username" => username,
    )
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_get_image(resource::Resource, sessionid::String, url::String)
    r = request(resource, "GET", url, Dict("sessionid" => sessionid))
    if r.status >= 400
        logstatus("animeplanet_get_image", r, url)
        return HTTP.Response(r)
    end
    ret = Dict("version" => API_VERSION, "url" => url, "image" => r.body)
    HTTP.Response(200, encode(ret, :msgpack)...)
end

function animeplanet_login(resource::Resource, sessionid::String)
    url = "https://www.anime-planet.com/api/login"
    creds = resource["credentials"]
    d = Dict(
        "_username" => creds["username"],
        "_password" => creds["password"],
        "_remember_me" => true,
    )
    headers, body = encode(d, :json)
    headers["sessionid"] = sessionid
    r = request(resource, "PUT", url, headers, body)
    if r.status >= 400
        logstatus("animeplanet_login", r, url; handled_errors = [])
        return HTTP.Response(r)
    end
    HTTP.Response(200, [])
end

function compile(::Integer) end
include("../julia_utils/start_oxygen.jl")

end
