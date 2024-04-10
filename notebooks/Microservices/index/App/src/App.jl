module App

import HTTP
import JSON
import MsgPack
import Oxygen

TEMPLATE = open(joinpath(@__DIR__, "index.html")) do f
    read(f, String)
end
ALL_MEDIUMS = ["manga", "anime"]
ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]
URLS = open(joinpath(@__DIR__, "environment/endpoints.json")) do f
    JSON.parse(read(f, String))
end

function msgpack(d::Dict)::HTTP.Response
    body = MsgPack.pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

function mock_response(app::String, path::String)
    if app == "ensemble" && startswith(path, "/query?")
        return Oxygen.text("success")
    else
        return msgpack(
            Dict(
                "alpha" => Dict("success" => true), 
                "features" => Dict("success" => true)
            )
        )
    end
end

function run(app::String, path::String, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response(app, path)
    else
        return HTTP.get(URLS[app] * path)
    end
end

function run(app::String, path::String, data::Vector{UInt8}, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response(app, path)
    else
        return HTTP.post(URLS[app] * path, [("Content-Type", "application/msgpack")], data)
    end
end

function get_media_lists(username::String, source::String, precompile::Bool)::Dict
    responses = Dict{Any,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        r1 = run(
            "fetch_media_lists",
            "/query?username=$username&source=$source&medium=$medium",
            precompile,
        )
        r2 = run(
            "compress_media_lists",
            "/query?username=$username&source=$source&medium=$medium",
            r1.body,
            precompile,
        )
        responses[medium] = MsgPack.unpack(r2.body)
    end
    responses
end

function nondirectional(data::Vector{UInt8}, precompile::Bool)::Dict
    r = run("nondirectional", "/query", data, precompile)
    MsgPack.unpack(r.body)
end

function bagofwords(data::Vector{UInt8}, precompile::Bool)::Dict
    r_process = run("bagofwords_jl", "/process", data, precompile)
    process = MsgPack.unpack(r_process.body)
    alphas = process["alpha"]
    input = MsgPack.pack(process["features"])

    responses = Dict{Any,Any}((x, y) => nothing for x in ALL_MEDIUMS for y in ALL_METRICS)
    @sync for metric in ALL_METRICS
        for medium in ALL_MEDIUMS
            Threads.@spawn begin
                r1 = run(
                    "bagofwords_py_$(medium)_$(metric)",
                    "/query?medium=$medium&metric=$metric",
                    input,
                    precompile,
                )
                d = Dict(
                    "embedding" => MsgPack.unpack(r1.body),
                    "payload" => MsgPack.unpack(data),
                )
                r2 = run(
                    "bagofwords_jl",
                    "/compute?medium=$medium&metric=$metric",
                    MsgPack.pack(d),
                    precompile,
                )
                responses[(medium, metric)] = MsgPack.unpack(r2.body)
            end
        end
    end
    merge(alphas, collect(values(responses))...)
end

function transformer(data::Vector{UInt8}, precompile::Bool)::Dict
    r_process = run("transformer_jl", "/process", data, precompile)
    input = r_process.body

    responses = Dict{String,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        r1 = run("transformer_py_$(medium)", "/query?medium=$medium", input, precompile)
        d = Dict(
            "embedding" => MsgPack.unpack(r1.body),
            "payload" => MsgPack.unpack(data),
        )
        r2 = run(
            "transformer_jl",
            "/compute?medium=$medium",
            MsgPack.pack(d),
            precompile,
        )
        responses[medium] = MsgPack.unpack(r2.body)
    end
    merge(collect(values(responses))...)
end

function ensemble(
    username::String,
    source::String,
    data::Vector{UInt8},
    responses::Dict,
    precompile::Bool,
)::String
    inputs = Dict(
        "payload" => MsgPack.unpack(data),
        "alphas" => merge(collect(values(responses))...),
    )
    r = run(
        "ensemble",
        "/query?username=$username&source=$source",
        MsgPack.pack(inputs),
        precompile,
    )
    String(r.body)
end

function get_recs(username::String, source::String, precompile::Bool)::String
    data = MsgPack.pack(get_media_lists(username, source, precompile))
    models = Dict(
        "bagofwords" => bagofwords,
        "transformer" => transformer,
        "nondirectional" => nondirectional,
    )
    responses = Dict{String,Any}(x => nothing for x in keys(models))
    Threads.@threads for k in collect(keys(models))
        responses[k] = models[k](data, precompile)
    end
    ensemble(username, source, data, responses, precompile)
end


wake(req::HTTP.Request) = msgpack(Dict("success" => true))
index(req::HTTP.Request) = Oxygen.html(TEMPLATE)

function heartbeat(req::HTTP.Request)
    form_data = HTTP.URIs.queryparams(String(req.body))
    precompile = get(form_data, "precompile", "") == "true"
    responses = Dict{Any,Any}(x => nothing for x in keys(URLS))
    Threads.@threads for x in collect(keys(URLS))
        if x == "index"
            continue
        end
        responses[x] = @elapsed run(x, "/wake", precompile)
    end
    msgpack(responses)
end

function submit(req::HTTP.Request)
    form_data = HTTP.URIs.queryparams(String(req.body))
    username = form_data["username"]
    source_map = Dict(
        "MyAnimeList" => "mal",
        "AniList" => "anilist",
        "Kitsu" => "kitsu",
        "Anime-Planet" => "animeplanet",
    )
    source = source_map[form_data["source"]]
    precompile = get(form_data, "precompile", "") == "true"
    recs = get_recs(username, source, precompile)
    Oxygen.html(recs)
end

function precompile_run(running::Bool, port::Int, query::String)
    if running
        return HTTP.get("http://localhost:$port$query")
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        r = HTTP.Request("GET", query, [], "")
        return fn(r)
    end
end

function precompile_run(running::Bool, port::Int, query::String, data::String)
    if running
        return HTTP.post(
            "http://localhost:$port$query",
            [("Content-Type", "application/x-www-form-urlencoded")],
            data,
        )
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        req = HTTP.Request("POST", query, [("Content-Type", "application/x-www-form-urlencoded")], data)
        return fn(req)
    end
end

function precompile(running::Bool, port::Int)
    while true
        try
            r = precompile_run(running, port, "/wake")
            if MsgPack.unpack(r.body)["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end

    precompile_run(running, port, "/heartbeat", "precompile=true")
    precompile_run(running, port, "/index")
    precompile_run(running, port, "/submit", "username=test&source=MyAnimeList&precompile=true")
end

end