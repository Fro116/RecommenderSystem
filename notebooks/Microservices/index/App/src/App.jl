module App

import HTTP
import JSON
import Oxygen

TEMPLATE = open(joinpath(@__DIR__, "index.html")) do f
    read(f, String)
end
ALL_MEDIUMS = ["manga", "anime"]
# TODO read from environment
URLS = Dict(
    "fetch_media_lists" => "http://fetch_media_lists:8080",
    "compress_media_lists" => "http://compress_media_lists:8080",
    "nondirectional" => "http://nondirectional:8080",
    "transformer_jl" => "http://transformer_jl:8080",
    "transformer_py" => "http://transformer_py:8080",
    "bagofwords_jl" => "http://bagofwords_jl:8080",
    "bagofwords_py" => "http://bagofwords_py:8080",
    "ensemble" => "http://ensemble:8080",
)

function mock_response()
    d = Dict(
        "alpha" => Dict("success" => true),
        "features" => Dict("success" => true),
    )
    HTTP.Response(JSON.json(d))
end

function run(app::String, path::String, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response()
    else
        HTTP.get(URLS[app] * path)
    end
end

function run(app::String, path::String, json::String, precompile::Bool)::HTTP.Response
    if precompile
        return mock_response()
    else
        HTTP.post(URLS[app] * path, [("Content-Type", "application/json")], json)
    end
end

function get_media_list(username::String, source::String, precompile::Bool)::String
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
    JSON.json(responses)
end

function transformer(data::String, precompile::Bool)::Dict
    r_process = run("transformer_jl", "/process", data, precompile)
    input = String(r_process.body)
    responses = Dict{Any,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for m in ALL_MEDIUMS
        responses[m] = run("transformer_py", "/query?medium=$m", input, precompile)
    end
    embeddings = merge([JSON.parse(String(copy(x.body))) for x in values(responses)]...)
    r_compute = run(
        "transformer_jl",
        "/compute",
        JSON.json(Dict("payload" => JSON.parse(data), "embeddings" => embeddings)),
        precompile,
    )
    JSON.parse(String(r_compute.body))
end

function bagofwords(data::String, precompile::Bool)::Dict
    r_process = run("bagofwords_jl", "/process", data, precompile)
    process = JSON.parse(String(r_process.body))
    alphas = process["alpha"]
    input = JSON.json(process["features"])
    responses = Dict{Any,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for m in ALL_MEDIUMS
        responses[m] = run("bagofwords_py", "/query?medium=$m", input, precompile)
    end
    embeddings = merge([JSON.parse(String(copy(x.body))) for x in values(responses)]...)
    r_compute = run(
        "bagofwords_jl",
        "/compute",
        JSON.json(Dict("payload" => JSON.parse(data), "embeddings" => embeddings)),
        precompile,
    )
    merge(alphas, JSON.parse(String(r_compute.body)))
end

function nondirectional(data::String, precompile::Bool)::Dict
    r = run("nondirectional", "/query", data, precompile)
    JSON.parse(String(r.body))
end

function get_recs(username::String, source::String, precompile::Bool)::String
    data = get_media_list(username, source, precompile)
    responses =
        Dict{Any,Any}(x => nothing for x in ["bagofwords", "transformer", "nondirectional"])
    @sync begin
        Threads.@spawn responses["bagofwords"] = bagofwords(data, precompile)
        Threads.@spawn responses["transformer"] = transformer(data, precompile)
        Threads.@spawn responses["nondirectional"] = nondirectional(data, precompile)
    end
    alphas = merge(values(responses)...)
    inputs = JSON.json(Dict("payload" => JSON.parse(data), "alphas" => alphas))
    r = run("ensemble", "/query?username=$username&source=$source", inputs, precompile)
    String(r.body)
end

wake(req::HTTP.Request) = Oxygen.json(Dict("success" => true))
index(req::HTTP.Request) = Oxygen.html(TEMPLATE)

function submit(req::HTTP.Request)
    form_data = HTTP.URIs.queryparams(String(req.body))
    username = form_data["username"]
    source_map = Dict(
        "MyAnimeList" => "mal",
        "AniList" => "anilist",
        "Kitsu" => "kitsu",
        "Anime-Planet" => "animeplanet",
        "precompile" => "precompile",
    )
    source = source_map[form_data["source"]]
    precompile = source == "precompile"
    recs = get_recs(username, source, precompile)
    Oxygen.html(recs)
end

function precompile(port::Int)
    while true
        try
            r = HTTP.get("http://0.0.0.0:$port/wake")
            json = JSON.parse(String(copy(r.body)))
            if json["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end

    HTTP.get("http://0.0.0.0:$port")
    HTTP.post(
        "http://0.0.0.0:$port/submit",
        [("Content-Type", "application/x-www-form-urlencoded")],
        "username=test&source=precompile",
    )
end

end