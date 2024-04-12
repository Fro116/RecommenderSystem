module App

import CodecZstd
import HTTP
import MsgPack
import Oxygen
import NBInclude: @nbinclude
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")

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

SIMILARITY_MATRICES = Dict(
    x => read_params(x, true)["S"] for medium in ALL_MEDIUMS for x in [
        "$medium/Nondirectional/RelatedSeries",
        "$medium/Nondirectional/RecapSeries",
        "$medium/Nondirectional/CrossRelatedSeries",
        "$medium/Nondirectional/CrossRecapSeries",
        "$medium/Nondirectional/SequelSeries",
        "$medium/Nondirectional/DirectSequelSeries",
    ]
)

read_similarity_matrix(outdir) = SIMILARITY_MATRICES[outdir]

function get_recommendee_split(medium, payload)
    df = get_raw_split(payload, medium, [:itemid, :status], nothing)
    invalid_statuses = get_status.([:plan_to_watch, :wont_watch, :none])
    filter(df, df.status .∉ (invalid_statuses,))
end

function compute_related_series_alpha(df, name, medium)
    x = zeros(Float32, num_items(medium))
    x[df.itemid.+1] .= 1
    S = read_similarity_matrix(name)
    S * x
end

function compute_cross_related_series_alpha(df, name, medium)
    if medium == "anime"
        cross_medium = "manga"
    elseif medium == "manga"
        cross_medium = "anime"
    else
        @assert false
    end
    x = zeros(Float32, num_items(cross_medium))
    x[df.itemid.+1] .= 1
    S = read_similarity_matrix(name)
    S' * x
end

function compute_sequel_series_alpha(df, name, medium)
    df = filter(df, df.status .>= get_status(:completed))
    x = zeros(Float32, num_items(medium))
    x[df.itemid.+1] .= 1
    S = read_similarity_matrix(name)
    S * x
end

function compute_dependency_alpha(df, name, outdir, medium)
    df = filter(df, df.status .>= get_status(:completed))
    x = ones(Float32, num_items(medium))
    x[df.itemid.+1] .= 0
    S = read_similarity_matrix(name)
    S * x
end

function wake(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

function query(req::HTTP.Request)
    payload = unpack(req.body)
    anime_df = get_recommendee_split("anime", payload)
    manga_df = get_recommendee_split("manga", payload)
    ret = Dict()
    for medium in ALL_MEDIUMS
        if medium == "anime"
            media_df = anime_df
            cross_media_df = manga_df
        elseif medium == "manga"
            media_df = manga_df
            cross_media_df = anime_df
        else
            @assert false
        end
        ret["$medium/Nondirectional/RelatedSeries"] = compute_related_series_alpha(
            media_df,
            "$medium/Nondirectional/RelatedSeries",
            medium,
        )
        ret["$medium/Nondirectional/RecapSeries"] = compute_related_series_alpha(
            media_df,
            "$medium/Nondirectional/RecapSeries",
            medium,
        )
        ret["$medium/Nondirectional/CrossRelatedSeries"] =
            compute_cross_related_series_alpha(
                cross_media_df,
                "$medium/Nondirectional/CrossRelatedSeries",
                medium,
            )
        ret["$medium/Nondirectional/CrossRecapSeries"] = compute_cross_related_series_alpha(
            cross_media_df,
            "$medium/Nondirectional/CrossRecapSeries",
            medium,
        )
        ret["$medium/Nondirectional/SequelSeries"] = compute_sequel_series_alpha(
            media_df,
            "$medium/Nondirectional/SequelSeries",
            medium,
        )
        ret["$medium/Nondirectional/DirectSequelSeries"] = compute_sequel_series_alpha(
            media_df,
            "$medium/Nondirectional/DirectSequelSeries",
            medium,
        )
        ret["$medium/Nondirectional/Dependencies"] = compute_dependency_alpha(
            media_df,
            "$medium/Nondirectional/SequelSeries",
            "$medium/Nondirectional/Dependencies",
            medium,
        )
    end
    msgpack(ret)
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

    payload = pack(
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
    HTTP.post(
        "http://localhost:$port/query",
        [("Content-Type", "application/msgpack")],
        payload,
    )
end

end
