module App

import CodecZstd
import CSV
import DataFrames
import Dates
import HTTP
import JSON
import MsgPack
import Oxygen
import NBInclude: @nbinclude
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")

include("./Linear.jl")
include("./Ranking.jl")
include("./Filter.jl")
include("./Render.jl")

TEMPLATE = open(joinpath(@__DIR__, "index.html")) do f
    read(f, String)
end
URLS = open(joinpath(@__DIR__, "environment/endpoints.json")) do f
    JSON.parse(read(f, String))
end

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

function mock_response(app::String, path::String)
    if path == "/wake"
        return msgpack(Dict("success" => true))
    elseif app in ["fetch_media_lists", "transformer_jl"]
        return msgpack(Dict("success" => true))
    elseif app == "compress_media_lists"
        m = findfirst(x -> x == split(path, "=")[end], ALL_MEDIUMS)
        return msgpack(
            Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[m],
                "priority" => Int32[0],
                "status" => Int32[6],
                "progress" => Float32[1.0],
                "updated_at" => Float32[1.0],
                "started_at" => Float32[0.0],
                "repeat_count" => Int32[0],
                "owned" => Int32[0],
                "sentiment" => Int32[0],
                "itemid" => Int32[0],
                "finished_at" => Float32[0.0],
                "source" => Int32[0],
                "userid" => Int32[0],
            )
        )
    elseif app == "nondirectional"
        d = Dict()
        for x in ALL_MEDIUMS
            for y in [
                "RelatedSeries",
                "SequelSeries",
                "CrossRelatedSeries",
                "CrossRecapSeries",
                "RecapSeries",
                "DirectSequelSeries",
                "Dependencies",
            ]
                d["$x/Nondirectional/$y"] = zeros(Float32, num_items(x))
            end
        end
        return msgpack(d)
    elseif app == "bagofwords_jl"
        return msgpack(
            Dict(
                "alphas" => Dict(
                    "$x/Baseline/rating" => zeros(Float32, num_items(x))
                    for x in ALL_MEDIUMS
                ),
                "dataset" => Dict("test" => [1])
            )
        )
    elseif startswith(app, "bagofwords_py")
        medium, metric = split(app, "_")[3:4]
        return msgpack(
            Dict("$medium/BagOfWords/v1/$metric" => zeros(Float32, num_items(medium)))
        )
    elseif startswith(app, "transformer_py")
        medium = split(app, "_")[3]
        d = Dict()
        for metric in ALL_METRICS
            d["$medium/Transformer/v1/$metric"] = zeros(Float32, num_items(medium))
        end
        return msgpack(d)                
    else
        @assert false
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
        responses[medium] = unpack(r2.body)
    end
    responses
end

function nondirectional(data::Vector{UInt8}, precompile::Bool)::Dict
    r = run("nondirectional", "/query", data, precompile)
    unpack(r.body)
end

function bagofwords(data::Vector{UInt8}, precompile::Bool)::Dict
    r_process = run("bagofwords_jl", "/query", data, precompile)
    process = unpack(r_process.body)
    alphas = process["alphas"]
    delete!(process, "alphas")
    input = pack(process)

    responses = Dict{Any,Any}((x, y) => nothing for x in ALL_MEDIUMS for y in ALL_METRICS)
    @sync for metric in ALL_METRICS
        for medium in ALL_MEDIUMS
            Threads.@spawn begin
                r = run(
                    "bagofwords_py_$(medium)_$(metric)",
                    "/query?medium=$medium&metric=$metric",
                    input,
                    precompile,
                )
                responses[(medium, metric)] = unpack(r.body)
            end
        end
    end
    merge(alphas, collect(values(responses))...)
end

function transformer(data::Vector{UInt8}, precompile::Bool)::Dict
    r_process = run("transformer_jl", "/query", data, precompile)
    input = r_process.body

    responses = Dict{String,Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for medium in ALL_MEDIUMS
        r = run("transformer_py_$(medium)", "/query?medium=$medium", input, precompile)
        responses[medium] = unpack(r.body)
    end
    merge(collect(values(responses))...)
end

function ensemble(
    username::String,
    source::String,
    data::Vector{UInt8},
    responses::Dict,
)::String
    payload = unpack(data)
    alphas = merge(collect(values(responses))...)
    linear = compute_linear(payload, alphas)
    alphas = merge(linear, alphas)
    d = Dict{String, Any}(x => nothing for x in ALL_MEDIUMS)
    Threads.@threads for x in ALL_MEDIUMS
        d[x] = recommend(payload, alphas, x, source)
    end
    render_html_page(username, d["anime"], d["manga"])
end

function get_recs(username::String, source::String, precompile::Bool)::String
    @time data = pack(get_media_lists(username, source, precompile))
    models = Dict(
        "bagofwords" => bagofwords,
        "transformer" => transformer,
        "nondirectional" => nondirectional,
    )
    responses = Dict{String,Any}(x => nothing for x in keys(models))
    @time Threads.@threads for k in collect(keys(models))
        responses[k] = models[k](data, precompile)
    end
    @time ensemble(username, source, data, responses)
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

function precompile(port::Int)
    awake = false
    while !awake
        try
            HTTP.get("http://localhost:$port/wake")
            awake = true
        catch
            @warn "service down"
            sleep(1)
        end
    end
    HTTP.get("http://localhost:$port/")
    HTTP.post(
        "http://localhost:$port/heartbeat",
        [("Content-Type", "application/x-www-form-urlencoded")],
        "precompile=true",
    )    
    HTTP.post(
        "http://localhost:$port/submit",
        [("Content-Type", "application/x-www-form-urlencoded")],
        "username=test&source=MyAnimeList&precompile=true",
    )        
end

end