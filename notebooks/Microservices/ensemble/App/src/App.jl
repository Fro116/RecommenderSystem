module App

import CSV
import DataFrames
import Dates
import HTTP
import JSON
import Oxygen
import NBInclude: @nbinclude
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")

include("./Linear.jl")
include("./Ranking.jl")
include("./Filter.jl")
include("./Render.jl")

function wake(req::HTTP.Request)
    Oxygen.json(Dict("success" => true))
end

function query(req::HTTP.Request)
    params = Oxygen.queryparams(req)
    username = params["username"]
    source = params["source"]
    data = JSON.parse(String(req.body))
    payload = data["payload"]
    alphas = data["alphas"]
    linear = compute_linear(payload, alphas)
    alphas = merge(linear, alphas)
    anime_recs = recommend(payload, alphas, "anime", source)
    manga_recs = recommend(payload, alphas, "manga", source)
    page = render_html_page(username, anime_recs, manga_recs)
    Oxygen.text(page)
end

function precompile(port::Int)
    while true
        try
            r = HTTP.get("http://localhost:$port/wake")
            json = JSON.parse(String(copy(r.body)))
            print(json)
            if json["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end

    payload = (
        "{\"anime\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0]," *
        "\"update_order\":[0],\"sentiment_score\":[0],\"medium\":[1],\"backward_order\":[1]," *
        "\"priority\":[0],\"progress\":[1.0],\"forward_order\":[1],\"status\":[6]," *
        "\"updated_at\":[1.0],\"started_at\":[0.0],\"repeat_count\":[0],\"owned\":[0]," *
        "\"sentiment\":[0],\"finished_at\":[0.0],\"source\":[0],\"unit\":[1],\"userid\":[0]}," *
        "\"manga\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0],\"update_order\":[0]," *
        "\"sentiment_score\":[0],\"medium\":[0],\"backward_order\":[1],\"priority\":[0]," *
        "\"progress\":[1.0],\"forward_order\":[1],\"status\":[6],\"updated_at\":[1.0]," *
        "\"started_at\":[0],\"repeat_count\":[0],\"owned\":[0],\"sentiment\":[0]," *
        "\"finished_at\":[0],\"source\":[0],\"unit\":[1],\"userid\":[0]}}"
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
    d = Dict("payload" => JSON.parse(payload), "alphas" => alphas)
    HTTP.post(
        "http://localhost:$port/query?username=user&source=mal",
        [("Content-Type", "application/json")],
        JSON.json(d),
    )
end

end
