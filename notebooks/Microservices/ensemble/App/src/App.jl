module App

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

function msgpack(d::Dict)::HTTP.Response
    body = MsgPack.pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

function wake(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

function query(req::HTTP.Request)
    params = Oxygen.queryparams(req)
    username = params["username"]
    source = params["source"]
    data = MsgPack.unpack(req.body)
    payload = data["payload"]
    alphas = data["alphas"]
    @time linear = compute_linear(payload, alphas)
    alphas = merge(linear, alphas)
    d = Dict{String, Any}(x => nothing for x in ALL_MEDIUMS)
    @time Threads.@threads for x in ALL_MEDIUMS
        d[x] = recommend(payload, alphas, x, source)
    end
    @time page = render_html_page(username, d["anime"], d["manga"])
    Oxygen.text(page)
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

function precompile_run(running::Bool, port::Int, query::String, data::Vector{UInt8})
    if running
        return HTTP.post(
            "http://localhost:$port$query",
            [("Content-Type", "application/msgpack")],
            data,
        )
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        req = HTTP.Request("POST", query, [("Content-Type", "application/msgpack")], data)
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
    
    payload = MsgPack.pack(
        Dict(
            "anime" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[1],
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
            ),
            "manga" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[0],
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
            ),
        ),
    )
    alpha_names = vcat(
        ["$x/BagOfWords/v1/$y" for x in ALL_MEDIUMS for y in ALL_METRICS],
        ["$x/Baseline/rating" for x in ALL_MEDIUMS],
        ["$x/Transformer/v1/$y" for x in ALL_MEDIUMS for y in ALL_METRICS],
        [
            "$x/Nondirectional/$y" for x in ALL_MEDIUMS for y in [
                "RelatedSeries",
                "SequelSeries",
                "CrossRelatedSeries",
                "CrossRecapSeries",
                "RecapSeries",
                "DirectSequelSeries",
                "Dependencies",
            ]
        ],
    )
    function dummy_value(x)
        if startswith(x, "anime")
            m = "anime"
        elseif startswith(x, "manga")
            m = "manga"
        else
            @assert false
        end
        if occursin("Nondirectional", x)
            return zeros(Int32, num_items(m))
        else
            return ones(Float32, num_items(m))
        end
    end
    alphas = Dict(x => dummy_value(x) for x in alpha_names)
    d = Dict("payload" => MsgPack.unpack(payload), "alphas" => alphas)
    precompile_run(running, port, "/query?username=user&source=mal", MsgPack.pack(d))
end

end
