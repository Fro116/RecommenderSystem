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

const registry = Dict(
    "fetch_media_lists" => 3001,
    "import_media_lists" => 3002,
    "nondirectional" => 3003,
    "bagofwords_jl" => 3004,
    "bagofwords_py_manga_rating" => 3005,
    "bagofwords_py_manga_watch" => 3006,
    "bagofwords_py_manga_plantowatch" => 3007,
    "bagofwords_py_manga_drop" => 3008,
    "bagofwords_py_anime_rating" => 3009,
    "bagofwords_py_anime_watch" => 3010,
    "bagofwords_py_anime_plantowatch" => 3011,
    "bagofwords_py_anime_drop" => 3012,
)

const PARAMS = Dict(
    (medium, metric) => read_params("linear/v1/streaming/$medium/$metric") for
    medium in ALL_MEDIUMS for metric in ALL_METRICS
)

READY::Bool = false

Oxygen.@get "/ready" function ready(req::HTTP.Request)
    if !READY
        return HTTP.Response(503, [])
    else
        return HTTP.Response(200, [])
    end
end

Oxygen.@get "/terminate" terminate(req::HTTP.Request) = Oxygen.terminate()

function curl(app::String, query::String)::HTTP.Response
    port = registry[app]
    HTTP.get("http://localhost:$port/$query")
end

function curl(app::String, query::String, data::Vector{UInt8})::HTTP.Response
    port = registry[app]
    headers = Dict(
        "Content-Type" => "application/msgpack",
    )
    HTTP.post("http://localhost:$port/$query", headers, data)
end

function fetch_media_lists(username, source, medium)
    inputs = let
        if source == "animeplanet"
            tasks = map(["user_media_data", "feed_data"]) do datatype
                Threads.@spawn curl(
                    "fetch_media_lists",
                    "query?username=$username&source=$source&medium=$medium&datatype=$datatype",
                )
            end
            media, feed = fetch.(tasks) .|> x -> unpack(x.body)
            feed_dict = Dict(k => v for (k, v) in zip(feed["url"], feed["updated_at"]))
            media["updated_at"] = [get(feed_dict, url, "") for url in media["url"]]
            pack(media)
        else
            curl(
                "fetch_media_lists",
                "query?username=$username&source=$source&medium=$medium",
            ).body
        end
    end
    r = curl(
        "import_media_lists",
        "query?username=$username&source=$source&medium=$medium",
        inputs,
    )
    Dict(medium => unpack(r.body))
end

function fetch_media_lists(username, source)
    tasks = map(ALL_MEDIUMS) do medium
        Threads.@spawn fetch_media_lists(username, source, medium)
    end
    pack(merge(fetch.(tasks)...))
end

function nondirectional(lists)
    unpack(curl("nondirectional", "query", lists).body)
end

function bagofwords(lists)
    d = unpack(curl("bagofwords_jl", "query", lists).body)
    inputs = pack(Dict("inputs" => d["inputs"]))
    delete!(d, "inputs")
    args = [(medium, metric) for medium in ALL_MEDIUMS for metric in ALL_METRICS]
    tasks = map(args) do (medium, metric)
        Threads.@spawn begin
            unpack(
                curl(
                    "bagofwords_py_$(medium)_$(metric)",
                    "query?medium=$medium&metric=$metric",
                    inputs,
                ).body,
            )
        end
    end
    merge(d, fetch.(tasks)...)
end

function ensemble(alphas, params, medium, metric)
    y = zeros(Float32, num_items(medium))
    for i = 1:length(params["alphas"])
        y += alphas[params["alphas"][i]] * params["β"][i]
    end
    if metric in ["watch", "plantowatch"]
        y += fill(1.0f0 / num_items(medium), length(y)) * params["β"][end]
    elseif metric == "drop"
        y += fill(1.0f0, length(y)) * params["β"][end-1]
        y += fill(0.0f0, length(y)) * params["β"][end]
    end
    y
end;

function ensemble(alphas, PARAMS)
    args = [(medium, metric) for medium in ALL_MEDIUMS for metric in ALL_METRICS]
    tasks = map(args) do (medium, metric)
        Threads.@spawn Dict(
            "linear/v1/streaming/$medium/$metric" =>
                ensemble(alphas, PARAMS[(medium, metric)], medium, metric),
        )
    end
    merge(fetch.(tasks)...)
end

Oxygen.@get "/query" function query(req::HTTP.Request)
    timetrace = [time()]
    if !READY && !HTTP.Messages.hasheader(req, "Startup")
        return HTTP.Response(503, [])
    end
    qp = Oxygen.queryparams(req)
    username, source = qp["username"], qp["source"]
    lists = fetch_media_lists(username, source)
    push!(timetrace, time())
    nondirectional_task = Threads.@spawn nondirectional(lists)
    alpha_tasks = [Threads.@spawn bagofwords(lists)]
    alphas = merge(fetch.(alpha_tasks)...)
    push!(timetrace, time())
    ens = ensemble(alphas, PARAMS)
    nd = fetch(nondirectional_task)
    ret = merge(nd, ens)
    push!(timetrace, time())
    ret["debug"] = Dict("timetrace" => timetrace[2:end] - timetrace[1:end-1])
    msgpack(ret)
end

function test_ready(port)
    running = false
    while !running
        try
            ret = HTTP.get("http://localhost:$port/ready"; status_exception = false)
            running = ret.status == 503
        catch e
            sleep(1)
        end
    end
end

function test_terminate(port)
    HTTP.get("http://localhost:$port/terminate")
end

function test_query(port)
    registry_copy = copy(registry)
    mock_port = 5000
    for k in keys(registry)
        registry[k] = mock_port
    end
    Threads.@spawn run(`julia mock.jl $mock_port`)
    test_ready(mock_port)
    HTTP.get(
        "http://localhost:$port/query?username=test&source=test",
        Dict("Startup" => true),
    )
    Threads.@spawn test_terminate(mock_port)
    for (k, v) in registry_copy
        registry[k] = v
    end
end

if isempty(ARGS)
    port = 8080
else
    port = parse(Int, ARGS[1])
end
Threads.@spawn begin
    test_ready(port)
    test_query(port)
    if isempty(ARGS)
        test_terminate(port)
    end
    global READY
    READY = true
end
Oxygen.serveparallel(; host="0.0.0.0", port=port, access_log=nothing)
