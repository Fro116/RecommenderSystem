module App

import HTTP
import JSON
import Oxygen

TEMPLATE = open(joinpath(@__DIR__, "index.html")) do f
    read(f, String)
end
ALL_MEDIUMS = ["manga", "anime"]
ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]
URLS = open(joinpath(@__DIR__, "environment/endpoints.json")) do f
    JSON.parse(read(f, String))
end

function mock_response()
    d = Dict("alpha" => Dict("success" => true), "features" => Dict("success" => true))
    HTTP.Response(JSON.json(d))
end

function run(app::String, path::String, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response()
    else
        return HTTP.get(URLS[app] * path)
    end
end

function run(app::String, path::String, json::String, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response()
    else
        return HTTP.post(URLS[app] * path, [("Content-Type", "application/json")], json)
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
            String(r1.body),
            precompile,
        )
        responses[medium] = JSON.parse(String(r2.body))
    end
    responses
end

function nondirectional(data::String, precompile::Bool)::Dict
    r = run("nondirectional", "/query", data, precompile)
    JSON.parse(String(r.body))
end

function bagofwords(data::String, precompile::Bool)::Dict
    r_process = run("bagofwords_jl", "/process", data, precompile)
    process = JSON.parse(String(copy(r_process.body)))
    alphas = process["alpha"]
    input = JSON.json(process["features"])

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
                d = Dict()
                d["embedding"] = JSON.parse(String(r1.body))
                d["payload"] = JSON.parse(data)
                r2 = run(
                    "bagofwords_jl",
                    "/compute?medium=$medium&metric=$metric",
                    JSON.json(d),
                    precompile,
                )
                responses[(medium, metric)] = JSON.parse(String(r2.body))
            end
        end
    end
    merge(alphas, collect(values(responses))...)
end

function transformer(data::String, precompile::Bool)::Dict
    r_process = run("transformer_jl", "/process", data, precompile)
    input = String(r_process.body)

    responses = Dict{Any,Any}((x, y) => nothing for x in ALL_MEDIUMS for y in ALL_METRICS)
    @sync for medium in ALL_MEDIUMS
        Threads.@spawn begin
            r1 = run("transformer_py_$(medium)", "/query?medium=$medium", input, precompile)
            embeddings = JSON.parse(String(r1.body))
            Threads.@spawn for metric in ALL_METRICS
                d = Dict()
                k = "$(medium)_$(metric)"
                d["embedding"] = Dict(k => get(embeddings, k, ""))
                d["payload"] = JSON.parse(data)
                r2 = run(
                    "transformer_jl",
                    "/compute?medium=$medium&metric=$metric",
                    JSON.json(d),
                    precompile,
                )
                responses[(medium, metric)] = JSON.parse(String(r2.body))
            end
        end
    end
    merge(collect(values(responses))...)
end

function ensemble(
    username::String,
    source::String,
    data::String,
    responses::Dict,
    precompile::Bool,
)::String
    inputs = Dict(
        "payload" => JSON.parse(data),
        "alphas" => merge(collect(values(responses))...),
    )
    r = run(
        "ensemble",
        "/query?username=$username&source=$source",
        JSON.json(inputs),
        precompile,
    )
    String(r.body)
end

function get_recs(username::String, source::String, precompile::Bool)::String
    data = JSON.json(get_media_lists(username, source, precompile))
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


wake(req::HTTP.Request) = Oxygen.json(Dict("success" => true))
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
    responses
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
            json = JSON.parse(String(copy(r.body)))
            if json["success"] == true
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