# Setup

const PORT = parse(Int, ARGS[1])
const RATELIMIT_WINDOW = parse(Int, ARGS[2])
const LAYER_1_URL = ARGS[3]
const DEFAULT_IMPERSONATE = ARGS[4]
const DEFAULT_TIMEOUT = parse(Int, ARGS[5])
const RESOURCE_PATH = ARGS[6]
const API_VERSION = "5.0.0"
const RAND_SALT = rand() # randomize sessionids

import CodecZstd
import CSV
import DataFrames
import Dates
import Glob
import HTTP
import JSON3
import Memoize: @memoize
import MsgPack
import Oxygen
import UUIDs


# Resources

function get_partition()
    # set this to (machine index, num machines) if running in a cluster
    (0, 1)
end;

const Resource = Dict{String,Any}

function load_resources()::Vector{Resource}
    credentials = Dict()

    # proxies
    proxies = []
    path = "$RESOURCE_PATH/proxies/geolocations.txt"
    if ispath(path)
        geo_df = CSV.read(path, DataFrames.DataFrame)
        valid_ips = Set(filter(x -> x["geo location"] == "us", geo_df).ip)
    else
        valid_ips = nothing
    end
    path = "$RESOURCE_PATH/proxies/proxies.txt"
    if ispath(path)
        proxy_df = CSV.read(
            path,
            DataFrames.DataFrame,
            header = ["host", "port", "username", "password"],
            delim = ':',
        )
        for (host, port, username, password) in
            zip(proxy_df.host, proxy_df.port, proxy_df.username, proxy_df.password)
            ip = split(username, "-")[end]
            if !isnothing(valid_ips) && ip ∉ valid_ips
                continue
            end
            push!(proxies, "http://$username:$password@$host:$port")
        end
    end
    proxies = sort(proxies)

    # mal (ip and token limit)
    mal_tokens =
        [only(readlines(x)) for x in Glob.glob("$RESOURCE_PATH/mal/authentication/*.txt")]
    mal_resources = [
        Dict("location" => "mal", "token" => x, "proxyurls" => [], "ratelimit" => 8) for
        x in mal_tokens
    ]
    i = 1
    for proxy in proxies
        if length(mal_resources[i]["proxyurls"]) < 10
            # bound token size
            push!(mal_resources[i]["proxyurls"], proxy)
        end
        i = ((i + 1) % length(mal_tokens)) + 1
    end

    # malweb (ip limit)
    malweb_resources =
        [Dict("location" => "malweb", "proxyurl" => x, "ratelimit" => 4) for x in proxies]

    # anilist (ip limit)
    anilist_resources =
        [Dict("location" => "anilist", "proxyurl" => x, "ratelimit" => 4) for x in proxies]

    # kitsu (ip limit)
    kitsu_credentials = []
    for x in Glob.glob("$RESOURCE_PATH/kitsu/authentication/*.txt")
        (username, password) = readlines(x)
        push!(kitsu_credentials, Dict("username" => username, "password" => password))
    end
    kitsu_resources = [
        Dict(
            "location" => "kitsu",
            "proxyurl" => x,
            "credentials" => kitsu_credentials,
            "ratelimit" => 4,
        ) for x in proxies
    ]

    # animeplanet (credit limit)
    animeplanet_token = only(readlines("$RESOURCE_PATH/scrapfly/key.txt"))
    animeplanet_concurrency =
        parse(Int, only(readlines("$RESOURCE_PATH/scrapfly/concurrency.txt")))
    animeplanet_resources = [
        Dict(
            "location" => "animeplanet",
            "token" => animeplanet_token,
            "uid" => uid,
            "ratelimit" => 8,
        ) for uid = 1:animeplanet_concurrency
    ]

    resources = vcat(
        mal_resources,
        malweb_resources,
        anilist_resources,
        kitsu_resources,
        animeplanet_resources,
    )
    # shard resources across multiple machines
    part, num_parts = get_partition()
    [x for (i, x) in Iterators.enumerate(resources) if (i % num_parts) == part]
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

function update_resources(r::Resources, refresh_secs::Real, timeout_secs::Real)
    while true
        sleep(refresh_secs)
        new_resources = Set(load_resources())
        old_resources = Set(keys(r.resources))
        lock(r.lock) do
            # update resources
            t = Dates.datetime2unix(Dates.now())
            for k in old_resources
                if k ∉ new_resources
                    # resource got assigned to a different machine
                    delete!(r.resources, k)
                    filter!(x -> x != k, r.index[k["location"]])
                else
                    m = r.resources[k]
                    if !isnothing(m.checkout_time) && t - m.checkout_time > timeout_secs
                        # resource was never returned, reclaim it
                        r.resources[k] =
                            ResourceMetadata(m.version + 1, nothing, m.request_times)
                        push_front!(r.index[k["location"]], k)
                    end
                end
            end
            for k in setdiff(new_resources, old_resources)
                # resource was added
                r.resources[k] = ResourceMetadata(0, nothing, [])
                push_front!(r.index[k["location"]], k)
            end
        end
    end
end

function take!(r::Resources, location::String, timeout::Real)
    start = Dates.datetime2unix(Dates.now())
    uuid = string(UUIDs.uuid4())
    lock(r.lock) do
        push!(RESOURCES.queue[location], uuid)
    end
    try
        while true
            val = lock(r.lock) do
                task_id = first(RESOURCES.queue[location])
                if task_id != uuid
                    return nothing
                end
                if isempty(r.index[location])
                    return nothing
                end
                k = popfirst!(r.index[location])
                m = r.resources[k]
                @assert isnothing(m.checkout_time)
                r.resources[k] = ResourceMetadata(
                    m.version,
                    Dates.datetime2unix(Dates.now()),
                    m.request_times,
                )
                Dict("resource" => k, "version" => m.version)
            end
            if !isnothing(val)
                return val
            end
            if Dates.datetime2unix(Dates.now()) - start > timeout
                return nothing
            end
            sleep(0.001)
        end
    finally
        lock(r.lock) do
            arr = RESOURCES.queue[location]
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

Threads.@spawn update_resources(RESOURCES, 60.0, 600.0);

function encode(d::Dict, encoding::Symbol)
    if encoding == :json
        headers = Dict("Content-Type" => "application/json")
        body = Vector{UInt8}(JSON3.write(d))
    elseif encoding == :msgpack
        headers = Dict("Content-Type" => "application/msgpack")
        body = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
    else
        @assert false
    end
    headers, body
end
function decode(r::HTTP.Message)::Dict
    if HTTP.headercontains(r, "Content-Type", "application/json")
        return JSON3.read(String(r.body), Dict{String,Any})
    elseif HTTP.headercontains(r, "Content-Type", "application/msgpack")
        return MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, r.body))
    else
        @assert false
    end
end

Oxygen.@post "/resources" function resources_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    if data["method"] == "take"
        token = take!(RESOURCES, data["location"], data["timeout"])
        if isnothing(token)
            return HTTP.Response(404, [])
        end
        return HTTP.Response(200, encode(token, :json)...)
    elseif data["method"] == "put"
        put!(RESOURCES, data["token"]["resource"], data["token"]["version"])
        return HTTP.Response(200, [])
    else
        @assert false
    end
end

function ratelimit!(x::ResourceMetadata, ratelimit::Real)
    window = RATELIMIT_WINDOW
    if !isempty(x.request_times)
        startindex = max(1, length(x.request_times) - window + 1)
        times = x.request_times[startindex:end]
        wait_until = first(times) + length(times) * ratelimit
        delta = wait_until - Dates.datetime2unix(Dates.now())
        if delta > 0
            sleep(delta)
        end
    end
    push!(x.request_times, Dates.datetime2unix(Dates.now()))
    if length(x.request_times) > window
        popfirst!(x.request_times)
    end
end

struct Response
    status::Int
    body::String
    headers::Dict{String,String}
end

function callproxy(
    method::String,
    url::String,
    headers::Dict{String,String},
    body::Union{Vector{UInt8},Nothing},
    proxyurl::Union{String,Nothing},
    sessionid::String,
)
    args = Dict{String,Any}("method" => method, "url" => url, "sessionid" => sessionid)
    if get(headers, "impersonate", DEFAULT_IMPERSONATE) == "true"
        args["impersonate"] = "chrome"
        delete!(headers, "impersonate")
    end
    args["timeout"] = get(headers, "timeout", DEFAULT_TIMEOUT)
    delete!(headers, "timeout")
    if !isnothing(body)
        @assert headers["Content-Type"] == "application/json"
        delete!(headers, "Content-Type")
        args["json"] = String(body)
    end
    if !isempty(headers)
        args["headers"] = headers
    end
    if !isnothing(proxyurl)
        args["proxyurl"] = proxyurl
    end
    r = HTTP.request("POST", LAYER_1_URL, encode(args, :json)..., status_exception = false)
    if r.status >= 400
        return Response(r.status, "", Dict())
    end
    data = decode(r)
    Response(
        data["status_code"],
        data["content"],
        Dict(lowercase(k) => v for (k, v) in data["headers"]),
    )
end

function request(
    resource::Resource,
    method::String,
    url::String,
    headers::Dict{String,String} = Dict(),
    body::Union{Vector{UInt8},Nothing} = nothing,
)::Response
    metadata = lock(RESOURCES.lock) do
        m = get(RESOURCES.resources, resource, nothing)
        if isnothing(m)
            return Response(500, "", Dict())
        end
        RESOURCES.resources[resource] = ResourceMetadata(
            m.version,
            Dates.datetime2unix(Dates.now()),
            m.request_times,
        )
    end
    ratelimit!(metadata, resource["ratelimit"])
    if resource["location"] in ["mal", "malweb", "anilist", "kitsu"]
        if "proxyurls" in keys(resource)
            proxyurl = rand(resource["proxyurls"])
        else
            proxyurl = resource["proxyurl"]
        end
        return callproxy(
            method,
            url,
            headers,
            body,
            proxyurl,
            string(hash((resource, proxyurl, RAND_SALT))),
        )
    elseif resource["location"] == "animeplanet"
        sessionid = string(hash((resource, RAND_SALT)))
        url = string(
            HTTP.URI(
                "https://api.scrapfly.io/scrape";
                query = Dict(
                    "session" => sessionid,
                    "key" => resource["token"],
                    "proxy_pool" => "public_datacenter_pool",
                    "url" => url,
                    "country" => "us",
                ),
            ),
        )
        return callproxy(method, url, headers, body, nothing, sessionid)
    else
        @assert false
    end
end

const STDOUT_LOCK = ReentrantLock()

function logerror(x::String)
    Threads.lock(STDOUT_LOCK) do
        println("$(Dates.now()) [ERROR] $x")
        flush(stdout)
    end
end;

@memoize function html_entity_map()
    Dict(
        String(k) => v["characters"] for (k, v) in JSON3.read(read("entities.json", String))
    )
end

# HTML parsing

function html_unescape(text::AbstractString)
    text = HTTP.unescapeuri(text)
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
    matches = Set(only(m.captures) for m in eachmatch(regex, text))
    if optional && isempty(matches)
        return nothing
    end
    if multiple
        return [strip(html_unescape(x)) for x in matches]
    end
    strip(html_unescape(only(matches)))
end

optget(x::AbstractDict, k::String) = get(x, k, nothing)

# MAL

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

function mal_time(x)
    if isnothing(x)
        return nothing
    end
    try
        return Dates.datetime2unix(
            Dates.DateTime(x, Dates.dateformat"yyyy-mm-ddTHH:MM:SS+00:00"),
        )
    catch
        logerror("mal_get_list could not parse time $x for $url")
        return nothing
    end
end

function mal_get_list(resource::Resource, username::String, medium::String, offset::Int)
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
        logerror("mal_get_list received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
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
            "uid" => x["node"]["id"],
            "status" => optget(ls, "status"),
            "score" => optget(ls, "score"),
            "progress" => optget(ls, progress_col),
            "progress_volumes" => optget(ls, "num_volumes_read"),
            "started_at" => optget(ls, "start_date"),
            "completed_at" => optget(ls, "finish_date"),
            "priority" => optget(ls, "priority"),
            "repeat" => optget(ls, "repeat_col"),
            "repeat_count" => optget(ls, repeat_count_col),
            "repeat_value" => optget(ls, repeat_value_col),
            "tags" => optget(ls, "tags"),
            "notes" => optget(ls, "comments"),
            "updated_at" => mal_time(optget(ls, "updated_at")),
        )
        push!(entries, d)
    end
    ret = Dict("data" => entries, "offset" => offset)
    if "next" in keys(json["paging"])
        ret["next"] = json["paging"]["next"]
    end
    HTTP.Response(200, encode(ret, :json)...)
end

function mal_get_media(resource::Resource, medium::String, itemid::Int)
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
        ],
        "anime" => [
            "status",
            "num_episodes",
            "start_season",
            "broadcast",
            "source",
            "average_episode_duration",
            "rating",
            "studios",
        ],
        "manga" => ["finished", "num_volumes", "num_chapters", "authors"],
    )
    params = Dict("fields" => join(vcat(fields["common"], fields[medium]), ","))
    url = string(HTTP.URI("https://api.myanimelist.net/v2/$medium/$itemid"; query = params))
    entries = []
    headers = Dict("X-MAL-CLIENT-ID" => resource["token"])
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logerror("mal_get_media received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    details = Dict(
        "version" => API_VERSION,
        "malid" => json["id"],
        "title" => json["title"],
        "synonyms" => optget(json["alternative_titles"], "synonyms"),
        "alttitles" => Dict(
            string(k) => v for
            (k, v) in json["alternative_titles"] if string(k) != "synonyms"
        ),
        "start_date" => optget(json, "start_date"),
        "end_date" => optget(json, "end_date"),
        "synopsis" => optget(json, "synopsis"),
        "genres" => [x["name"] for x in json["genres"]],
        "created_at" => mal_time(optget(json, "created_at")),
        "updated_at" => mal_time(optget(json, "updated_at")),
        "mediatype" => json["media_type"],
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
        "pgrating" => optget(json, "rating"),
        "studios" =>
            "studios" in keys(json) ? [x["name"] for x in json["studios"]] : nothing,
        "num_volumes" => optget(json, "num_volumes"),
        "num_chapters" => optget(json, "num_chapters"),
        "authors" =>
            "authors" in keys(json) ?
            [Dict("id" => x["node"]["id"], "role" => x["role"]) for x in json["authors"]] : nothing,
    )
    # the mal API does not return manga relations for anime entries and vice versa        
    relations = nothing
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :json)...)
end

# MAL Web

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

function malweb_get_username(resource::Resource, userid::Int)
    url = "https://myanimelist.net/comments.php?id=$userid"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("malweb_get_username received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    for m in eachmatch(r"/profile/([^\"/%]+)\"", r.body)
        username = html_unescape(only(m.captures))
        ret = Dict("version" => API_VERSION, "username" => username)
        return HTTP.Response(200, encode(ret, :json)...)
    end
    HTTP.Response(404, [])
end

function malweb_get_media(resource::Resource, medium::String, itemid::Int)
    url = "https://myanimelist.net/$medium/$itemid"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("malweb_get_media status code $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end
    relations = malweb_get_media_relations(r.body, medium, itemid)
    ret = Dict("details" => nothing, "relations" => relations)
    HTTP.Response(200, encode(ret, :json)...)
end

function malweb_get_media_relations(text::String, medium::String, itemid::Int)
    relation_types = Dict(
        "Sequel" => "SEQUEL",
        "Prequel" => "PREQUEL",
        "Alternative Setting" => "ALTERNATIVE_SETTING",
        "Alternative Version" => "ALTERNATIVE_VERSION",
        "Side Story" => "SIDE_STORY",
        "Summary" => "SUMMARY",
        "Full Story" => "FULL_STORY",
        "Parent Story" => "PARENT_STORY",
        "Spin-Off" => "SPIN_OFF",
        "Adaptation" => "ADAPTATION",
        "Character" => "CHARACTER",
        "Other" => "OTHER",
    )
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
                    "malweb_get_media_relations did not finish parsing $last_href for $medium $itemid",
                )
            end
            return collect(records)
        end
        if prev_line == """<div class="relation">"""
            line = strip(first(split(line, "\n")))
            last_relation = get(relation_types, line, nothing)
            if isnothing(last_relation)
                logerror(
                    "malweb_get_media_relations could not parse relation $line for $medium $itemid",
                )
                continue
            end
            continue
        end
        if prev_line == """<td valign="top" class="ar fw-n borderClass nowrap">"""
            picture_section = false
            line = line[1:end-1] # strip trailing colon
            last_relation = get(relation_types, line, nothing)
            if isnothing(last_relation)
                logerror(
                    "malweb_get_media_relations could not parse relation $line for $medium $itemid",
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
                        "malweb_get_media_relations unexpected href $line for $medium $itemid",
                    )
                    last_href = nothing
                    continue
                end
            end
            if isnothing(last_relation)
                logerror(
                    "malweb_get_media_relations could not find relation for $line for $medium $itemid",
                )
                continue
            end
            m_medium, m_itemid = m.captures
            d = Dict(
                "version" => API_VERSION,
                "relation" => last_relation,
                "source_id" => itemid,
                "source_medium" => medium,
                "target_id" => parse(Int, m_itemid),
                "target_medium" => m_medium,
            )
            push!(records, d)
            continue
        end
    end
    logerror("malweb_get_media_relations could not parse relations $medium $itemid")
    collect(records)
end

function malweb_get_user(resource::Resource, username::String)
    url = "https://myanimelist.net/profile/$username"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("malweb_get_media status code $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end

    function info_panel(field)
        extract(
            r.body,
            """<span class="user-status-title di-ib fl-l fw-b">$field</span><span class="user-status-data di-ib fl-r">""",
            "</span>";
            optional = true,
        )
    end

    function info_links(field)
        parse(
            Int,
            extract(
                r.body,
                """<span class="user-status-title di-ib fl-l fw-b">$field</span><span class="user-status-data di-ib fl-r fw-b">""",
                "</span>";
            ),
        )
    end

    function item_counts(field)
        parse(
            Int,
            extract(
                r.body,
                """/$field/.*?<span class="di-ib fl-l fn-grey2">Total Entries</span><span class="di-ib fl-r">""",
                "</span>",
            ),
        )
    end

    ret = Dict(
        "version" => API_VERSION,
        "username" => username,
        "userid" =>
            extract(
                r.body,
                """href="https://myanimelist.net/modules.php\\?go=report&amp;type=profile&amp;id=""",
                "\"",
            ) |> x -> parse(Int, x),
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
    )
    HTTP.Response(200, encode(ret, :json)...)
end

# Anilist

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
        elseif endpoint == "media"
            return anilist_get_media(resource, data["medium"], data["itemid"])
        elseif endpoint == "userid"
            return anilist_get_userid(resource, data["username"])
        elseif endpoint == "user"
            return anilist_get_user(resource, data["userid"])
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

function anilist_date(x)
    function getd(x, key, default)
        y = get(x, key, default)
        if isnothing(y)
            return default
        end
        y
    end
    string(
        getd(x, "year", ""),
        "-",
        getd(x, "month", ""),
        "-",
        getd(x, "date", getd(x, "day", "")),
    )
end

function anilist_get_list(resource::Resource, userid::Int, medium::String, chunk::Int)
    url = "https://graphql.anilist.co"
    query = """
    query (\$userID: Int, \$MEDIA: MediaType, \$chunk: Int, \$perChunk: Int) {
        MediaListCollection (userId: \$userID, type: \$MEDIA, chunk: \$chunk, perChunk: \$perChunk) {
            hasNextChunk
            user {
                name
            }        
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
        logerror("anilist_get_list received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    username = json["data"]["MediaListCollection"]["user"]["name"]
    entries = Dict()
    for x in json["data"]["MediaListCollection"]["lists"]
        for entry in x["entries"]
            key = entry["mediaId"]
            if key ∉ keys(entries)
                entries[key] = Dict(
                    "version" => API_VERSION,
                    "userid" => userid,
                    "username" => username,
                    "anilistid" => entry["mediaId"],
                    "status" => entry["status"],
                    "score" => entry["score"],
                    "progress" => entry["progress"],
                    "progress_volumes" => entry["progressVolumes"],
                    "repeat" => entry["repeat"],
                    "priority" => entry["priority"],
                    "private" => entry["private"],
                    "notes" => entry["notes"],
                    "listnames" => String[],
                    "advancedScores" => entry["advancedScores"],
                    "started_at" => anilist_date(entry["startedAt"]),
                    "completed_at" => anilist_date(entry["completedAt"]),
                    "updated_at" => entry["updatedAt"],
                    "created_at" => entry["createdAt"],
                )
            end
            if x["isCustomList"]
                push!(entries[key]["listnames"], x["name"])
            end
        end
    end
    has_next_chunk = json["data"]["MediaListCollection"]["hasNextChunk"]
    ret =
        Dict("data" => collect(values(entries)), "chunk" => chunk, "next" => has_next_chunk)
    HTTP.Response(200, encode(ret, :json)...)
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
        return HTTP.Response(r.status, [])
    end
    ret =
        Dict("version" => API_VERSION, "userid" => JSON3.read(r.body)["data"]["User"]["id"])
    HTTP.Response(200, encode(ret, :json)...)
end

function anilist_get_media(resource::Resource, medium::String, itemid::Int)
    url = "https://graphql.anilist.co"
    query = """
    query (\$id: Int, \$MEDIA: MediaType)
    {
        Media (id: \$id, type:\$MEDIA) {
            id
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
            characters(sort: RELEVANCE) {
                edges {
                    role
                    node {
                        name {
                            full
                        }
                    }
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
            reviews {
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
            }
            recommendations(sort: RATING_DESC) {
                nodes {
                    rating
                    mediaRecommendation {
                        id
                        type
                    }
                }
            }
            isRecommendationBlocked
            isReviewBlocked
            modNotes
        }
    }"""
    variables = Dict("id" => itemid, "MEDIA" => uppercase(medium))
    r = request(
        resource,
        "POST",
        url,
        encode(Dict("query" => query, "variables" => variables), :json)...,
    )
    if r.status >= 400
        logerror("anilist_get_media received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end

    function season(year, season)
        if isnothing(year) || isnothing(season)
            return nothing
        end
        "$year $season"
    end
    json = JSON3.read(r.body)
    data = json["data"]["Media"]
    details = Dict(
        "version" => API_VERSION,
        "anilistid" => data["id"],
        "title" => optget(data["title"], "romaji"),
        "english_title" => optget(data["title"], "english"),
        "native_title" => optget(data["title"], "native"),
        "mediatype" => optget(data, "format"),
        "status" => optget(data, "status"),
        "summary" => optget(data, "description"),
        "startdate" => anilist_date(optget(data, "startDate")),
        "enddate" => anilist_date(optget(data, "endDate")),
        "season" => season(optget(data, "seasonYear"), optget(data, "season")),
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
            Dict("role" => x["role"], "name" => x["node"]["name"]["full"]) for
            x in get(data["characters"], "edges", [])
        ],
        "staffPeek" => [
            Dict("role" => x["role"], "name" => x["node"]["name"]["full"]) for
            x in get(data["staff"], "edges", [])
        ],
        "studios" => [x["name"] for x in get(data["studios"], "nodes", [])],
        "externalUrls" => [x["url"] for x in get(data["studios"], "externalLinks", [])],
        "isAdult" => optget(data, "isAdult"),
        "reviewsPeek" => get(data["reviews"], "nodes", []),
        "recommendationsPeek" => [
            Dict(
                "rating" => x["rating"],
                "medium" => x["mediaRecommendation"]["type"],
                "anilistid" => x["mediaRecommendation"]["id"],
            ) for x in get(data["recommendations"], "nodes", [])
        ],
        "isRecommendationBlocked" => optget(data, "isRecommendationBlocked"),
        "isReviewBlocked" => optget(data, "isReviewBlocked"),
        "modNotes" => optget(data, "modNotes"),
    )
    relations = []
    for e in get(data["relations"], "edges", [])
        d = Dict(
            "version" => API_VERSION,
            "relation" => e["relationType"],
            "source_id" => itemid,
            "source_media" => medium,
            "target_id" => e["node"]["id"],
            "target_media" => e["node"]["type"],
        )
        push!(relations, d)
    end
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :json)...)
end

function anilist_get_user(resource::Resource, userid::Int)
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
        logerror("anilist_get_media received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
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
    HTTP.Response(200, encode(ret, :json)...)
end

# Kitsu

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
            return kitsu_get_list(
                resource,
                data["auth"],
                data["userid"],
                data["medium"],
                data["offset"],
            )
        elseif endpoint == "media"
            return kitsu_get_media(resource, data["auth"], data["medium"], data["itemid"])
        elseif endpoint == "userid"
            return kitsu_get_userid(resource, data["auth"], data["username"], data["key"])
        elseif endpoint == "user"
            return kitsu_get_user(resource, data["auth"], data["userid"])
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

function kitsu_time(x)
    if isnothing(x)
        return nothing
    end
    try
        return Dates.datetime2unix(
            Dates.DateTime(x, Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        )
    catch
        logerror("kitsu_get_list could not parse time $x for $url")
        return nothing
    end
end

function kitsu_get_token(resource::Resource)
    credentials = rand(resource["credentials"])
    username = credentials["username"]
    password = credentials["password"]
    url = "https://kitsu.app/api/oauth/token"
    body = Dict("grant_type" => "password", "username" => username, "password" => password)
    headers, content = encode(body, :json)
    headers["impersonate"] = "true"
    r = request(resource, "POST", url, headers, content)
    if r.status >= 400
        logerror("kitsu_get_token received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    data = JSON3.read(r.body)
    token = data["access_token"]
    expires_in = data["expires_in"]
    expiry_time = Dates.datetime2unix(Dates.now()) + expires_in
    ret = Dict("token" => token, "expiry_time" => expiry_time)
    HTTP.Response(200, encode(ret, :json)...)
end

function kitsu_get_userid(resource::Resource, auth::String, username::String, key::String)
    url = string(
        HTTP.URI(
            "https://kitsu.app/api/edge/users";
            query = Dict("filter[$key]" => username),
        ),
    )
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => "true")
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logerror("kitsu_get_userid received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    data = JSON3.read(r.body)["data"]
    if length(data) != 1
        return HTTP.Response(404, [])
    end
    ret = Dict("version" => API_VERSION, "userid" => parse(Int, only(data)["id"]))
    HTTP.Response(200, encode(ret, :json)...)
end

function kitsu_get_media(resource::Resource, auth::String, medium::String, itemid::Int)
    params = Dict(
        "include" => "genres,mappings,mediaRelationships.destination",
        "fields[genres]" => "name",
        "fields[mappings]" => "externalSite,externalId",
    )
    url = string(HTTP.URI("https://kitsu.app/api/edge/$medium/$itemid"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => "true")
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logerror("kitsu_get_media received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    data = json["data"]["attributes"]
    function extractid(x)
        if isempty(x)
            return nothing
        end
        parse(Int, first(x))
    end
    details = Dict(
        "version" => API_VERSION,
        "kitsuid" => itemid,
        "createdAt" => kitsu_time(data["createdAt"]),
        "updatedAt" => kitsu_time(data["updatedAt"]),
        "summary" => data["synopsis"],
        "alttitles" => data["titles"],
        "title" => data["canonicalTitle"],
        "startdate" => data["startDate"],
        "enddate" => data["endDate"],
        "pgrating" => data["ageRating"],
        "ageRatingGuide" => data["ageRatingGuide"],
        "type" => data["subtype"],
        "status" => data["status"],
        "posterImage" => Dict(k => v for (k, v) in data["posterImage"] if k != :meta),
        "coverImage" => Dict(k => v for (k, v) in data["posterImage"] if k != :meta),
        "episodes" => optget(data, "episodeCount"),
        "duration" => optget(data, "episodeLength"),
        "chapters" => optget(data, "chapterCount"),
        "volumes" => optget(data, "volumeCount"),
        "youtubeVideoId" => optget(data, "youtubeVideoId"),
        "nsfw" => optget(data, "nsfw"),
        "genres" => [
            x["attributes"]["name"] for x in json["included"] if x["type"] == "genres"
        ],
        "malid" => extractid([
            x["attributes"]["externalId"] for
            x in json["included"] if x["type"] == "mappings" &&
            startswith(x["attributes"]["externalSite"], "myanimelist")
        ]),
        "anilistid" => extractid([
            x["attributes"]["externalId"] for
            x in json["included"] if x["type"] == "mappings" &&
            startswith(x["attributes"]["externalSite"], "anilist")
        ]),
    )

    relations = []
    for x in filter(r -> r["type"] == "mediaRelationships", json["included"])
        d = Dict(
            "version" => API_VERSION,
            "relation" => x["attributes"]["role"],
            "source_id" => itemid,
            "source_media" => medium,
            "target_id" => parse(Int, x["relationships"]["destination"]["data"]["id"]),
            "target_media" => x["relationships"]["destination"]["data"]["type"],
        )
        push!(relations, d)
    end
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :json)...)
end

function kitsu_get_list(
    resource::Resource,
    auth::String,
    userid::Int,
    medium::String,
    offset::Int,
)
    url = "https://kitsu.io/api/edge/library-entries"
    params = Dict(
        "fields[libraryEntries]" => join(
            [
                "ratingTwenty",
                "rating",
                "status",
                "progress",
                "volumesOwned",
                "reconsuming",
                "reconsumeCount",
                "notes",
                "private",
                "reactionSkipped",
                "progressedAt",
                "updatedAt",
                "createdAt",
                "startedAt",
                "finishedAt",
            ],
            ",",
        ),
        "fields[users]" => "name,slug",
        "include" => "user",
        "filter[user_id]" => userid,
        "filter[kind]" => medium,
        "page[limit]" => 500,
    )
    if offset > 0
        params["page[offset]"] = offset
    end
    url = string(HTTP.URI("https://kitsu.app/api/edge/library-entries"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => "true")
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logerror("kitsu_get_media received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    kitsu_rating(x::Nothing) = nothing
    kitsu_rating(x::String) = parse(Float64, x)
    kitsu_rating(x::Real) = convert(Float64, x)
    entries = []
    for x in json["data"]
        d = Dict(
            "version" => API_VERSION,
            "userid" => userid,
            "username" => only(json["included"])["attributes"]["name"],
            "userslug" => only(json["included"])["attributes"]["slug"],
            "kitsuid" => parse(Int, x["id"]),
            "status" => x["attributes"]["status"],
            "progress" => x["attributes"]["progress"],
            "progress_volumes" => optget(x["attributes"], "volumesOwned"),
            "repeat" => x["attributes"]["reconsuming"],
            "repeat_count" => x["attributes"]["reconsumeCount"],
            "notes" => x["attributes"]["notes"],
            "private" => x["attributes"]["private"],
            "updatedAt" => kitsu_time(optget(x["attributes"], "updatedAt")),
            "created_at" => kitsu_time(optget(x["attributes"], "created_at")),
            "started_at" => kitsu_time(optget(x["attributes"], "started_at")),
            "completed_at" => kitsu_time(optget(x["attributes"], "completed_at")),
            "progressedAt" => kitsu_time(optget(x["attributes"], "progressedAt")),
            "reactionSkipped" => x["attributes"]["reactionSkipped"],
            "ratingTwenty" => kitsu_rating(optget(x["attributes"], "ratingTwenty")),
            "rating" => kitsu_rating(optget(x["attributes"], "rating")),
        )
        push!(entries, d)
    end
    ret = Dict("data" => entries, "offset" => offset)
    if "next" in keys(json["links"])
        ret["next"] = json["links"]["next"]
    end
    HTTP.Response(200, encode(ret, :json)...)
end

function kitsu_get_user(resource::Resource, auth::String, userid::Int)
    params = Dict("include" => "stats")
    url = string(HTTP.URI("https://kitsu.app/api/edge/users/$userid"; query = params))
    headers = Dict("Authorization" => "Bearer $auth", "impersonate" => "true")
    r = request(resource, "GET", url, headers)
    if r.status >= 400
        logerror("kitsu_get_user received status $(r.status) $(r.body) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    data = json["data"]["attributes"]
    included = json["included"]

    ret = Dict(
        "version" => API_VERSION,
        "createdAt" => kitsu_time(data["createdAt"]),
        "updatedAt" => kitsu_time(data["updatedAt"]),
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
        "manga_counts" => only([
            x for x in included if x["attributes"]["kind"] == "manga-amount-consumed"
        ])["attributes"]["statsData"]["media"],
        "anime_counts" => only([
            x for x in included if x["attributes"]["kind"] == "anime-amount-consumed"
        ])["attributes"]["statsData"]["media"],
    )
    HTTP.Response(200, encode(ret, :json)...)
end

# Animeplanet

Oxygen.@post "/animeplanet" function animeplanet_api(r::HTTP.Request)::HTTP.Response
    data = decode(r)
    endpoint = data["endpoint"]
    token = data["token"]
    resource = token["resource"]
    if resource["location"] != "animeplanet"
        logerror("""animeplanet_api invalid resource $(resource["location"])""")
        return HTTP.Response(500, [])
    end
    try
        if endpoint == "list"
            return animeplanet_get_list(
                resource,
                data["medium"],
                data["username"],
                data["page"],
                data["expand_pagelimit"],
            )
        elseif endpoint == "media"
            return animeplanet_get_media(resource, data["medium"], data["itemid"])
        elseif endpoint == "user"
            return animeplanet_get_user(resource, data["username"])
        elseif endpoint == "username"
            return animeplanet_get_username(resource, data["userid"])
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

function animeplanet_get_list(
    resource::Resource,
    medium::String,
    username::String,
    page::String,
    expand_pagelimit::Bool,
)
    params = Dict("sort" => "user_updated", "order" => "desc")
    if page != 1
        params["page"] = page
    end
    if expand_pagelimit
        params["per_page"] = 560
    end
    url = string(
        HTTP.URI("https://www.anime-planet.com/users/$username/$medium"; query = params),
    )
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    if !json["result"]["success"]
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(500, [])
    end
    if occursin("has chosen to make their content private.", r.body)
        return HTTP.Response(403, [])
    end
    if occursin("<title>Search Results for $username", r.body)
        # invalid username
        return HTTP.Response(404, [])
    end
    default_pagelimit =
        occursin("""<option value="35" selected="selected">35</option>""", text)
    page_numbers = Set([
        parse(Int, only(m.captures)) for m in eachmatch(r"""&amp;page=([0-9]*)'>""", text)
    ])
    if default_pagelimit && page + 1 in page_numbers
        ret = Dict("next_page" => true, "extend_pagelimit" => true)
        return HTTP.Response(200, encode(ret, :json)...)
    end
    text = json.result.content

    function get_score(line)
        matches = extract(line, """<div class='ttRating'>""", "</div>", multiple = true)
        if isempty(matches)
            return nothing
        end
        2 * parse(Float64, only(matches))
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

    entries = []
    prevline = nothing
    for line in split(text, "\n")
        if occursin("<h3 class='cardName' >", line)
            d = Dict(
                "version" => API_VERSION,
                "username" => username,
                "title" =>
                    html_unescape(extract(line, """<h3 class='cardName' >""", "</h3>")),
                "url" => html_unescape(extract(prevline, """href="/$medium/""", '"')),
                "score" => get_score(line),
                "status" => parse(
                    Int,
                    html_unescape(extract(line, """<span class='status""", "'>")),
                ),
                "progress" => get_progress(line),
                "item_order" => nothing,
            )
            push!(entries, d)
        end
        prevline = line
    end
    ret = Dict("data" => entries)
    HTTP.Response(200, encode(ret, :json)...)
end

function animeplanet_get_media(resource::Resource, medium::String, itemname::String)
    url = "https://www.anime-planet.com/$medium/$itemname"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    if !json["result"]["success"]
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(500, [])
    end
    text = json.result.content

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
        "url" => itemname,
        "title" =>
            extract(text, """<h1 itemprop="name".*?>""", "</h1>", optional = true),
        "alttitle" => extract(
            text,
            """<h2 class="aka">(?s).*?Alt title:""",
            "</h2>",
            optional = true,
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
            "source_id" => itemname,
            "source_media" => medium,
            "target_id" => m.captures[2],
            "target_media" => m.captures[1],
        ) for m in eachmatch(
            Regex("""<a href="/(manga|anime)/(.*?)\" class="RelatedEntry"""),
            text,
        )
    ]
    ret = Dict("details" => details, "relations" => relations)
    HTTP.Response(200, encode(ret, :json)...)
end

function animeplanet_get_user(resource::Resource, username::String)
    url = "https://www.anime-planet.com/users/$username"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    if !json["result"]["success"]
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(500, [])
    end
    text = json.result.content
    has_profile = occursin(
        """<meta name="description" content="Meet $username on Anime-Planet.""",
        text,
    )
    if !has_profile
        return HTTP.Response(403, [])
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
    ret = Dict(
        "about" =>
            extract(text, """<section class="profBio userContent">""", "</section>"),
        "joined" =>
            extract(text, """<i class="fa fa-calendar"></i>""", "<", optional = true),
        "age" => extract(text, """<i class="fa fa-user"></i>""", "<", optional = true),
        "location" =>
            extract(text, """<i class="fa fa-home"></i>""", "<", optional = true),
        "anime_count" =>
            isempty(anime_counts) ? 0 : parse(Int, replace(first(anime_counts), "," => "")),
        "manga_count" =>
            isempty(manga_counts) ? 0 : parse(Int, replace(first(manga_counts), "," => "")),
        "last_online" => extract(
            text,
            """<h1 id="profileName" class="tooltip" title=\"""",
            "\"",
            optional = true,
        ),
        "followers" => parse(Int, extract(text, """followers">""", " Followers<")),
        "following" => parse(Int, extract(text, """following">""", " Following<")),
    )
    HTTP.Response(200, encode(ret, :json)...)
end

function animeplanet_get_username(resource::Resource, userid::Int)
    url = "https://www.anime-planet.com/forum/members/$userid"
    r = request(resource, "GET", url, Dict{String,String}())
    if r.status >= 400
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(r.status, [])
    end
    json = JSON3.read(r.body)
    if !json["result"]["success"]
        if occursin("Oops! We ran into some problems", json.result.content)
            # user does not exist
            return HTTP.Response(404, [])
        end
        logerror("animeplanet_get_media received status $(r.status) for $url")
        return HTTP.Response(500, [])
    end
    text = json.result.content
    ret = Dict(
        "userid" => userid,
        "username" => extract(
            text,
            "https://www.anime-planet.com/forum/members/",
            ".$userid",
            capture = "([a-zA-Z0-9_]+?)",
        ),
    )
end

Oxygen.serveparallel(; host = "0.0.0.0", port = PORT, access_log = nothing)