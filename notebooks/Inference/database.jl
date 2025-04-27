module database

import CSV
import DataFrames
import Oxygen
include("../julia_utils/http.jl")
include("../julia_utils/database.jl")
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
include("database_internals.jl")

const PORT = parse(Int, ARGS[1])
const LAYER_3_URL = ARGS[2] != "nothing" ? ARGS[2] : nothing
const EMBED_URL = ARGS[3] != "nothing" ? ARGS[3] : nothing

function get_fingerprints(source::String, items::DataFrames.DataFrame)
    fingerprints = []
    if source in ["mal", "anilist"]
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) == 0
                continue
            end
            if source == "mal"
                update_col = "updated_at"
            elseif source == "anilist"
                update_col = "updatedat"
            else
                @assert false
            end
            d = last(sort(sdf, update_col))
            d = Dict(
                "version" => d["version"],
                "medium" => d["medium"],
                "itemid" => d["itemid"],
                "updated_at" => d[update_col],
            )
            push!(fingerprints, d)
        end
    elseif source == "kitsu"
        if DataFrames.nrow(items) != 0
            update_col = "updatedat"
            d = last(sort(items, update_col))
            d = Dict("version" => d["version"], "updated_at" => d[update_col])
            push!(fingerprints, d)
        end
    elseif source == "animeplanet"
        for m in ["manga", "anime"]
            sdf = filter(x -> x.medium == m, items)
            if DataFrames.nrow(sdf) != 0
                d = last(sort(sdf, :item_order))
                d = Dict(
                    "version" => d["version"],
                    "medium" => d["medium"],
                    "itemid" => d["itemid"],
                )
                push!(fingerprints, d)
            end
            d = Dict("$(m)_count" => DataFrames.nrow(sdf))
            push!(fingerprints, d)
        end
    end
    fingerprints
end

function normalize(d::AbstractDict)
    norm(x::AbstractDict) = JSON3.write(x)
    norm(x::AbstractVector) = JSON3.write(x)
    norm(::Nothing) = missing
    norm(x) = x
    Dict(lowercase(k) => norm(v) for (k, v) in d)
end
normalize(d::AbstractVector) = [normalize(x) for x in d]

Oxygen.@post "/fetch_user_history" fetch_user_history(r::HTTP.Request)::HTTP.Response = fetch_user_history(decode(r))
function fetch_user_history(data::Dict)::HTTP.Response
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    source = data["source"]
    username = data["username"]
    if source in ["mal", "animeplanet"]
        args = Dict("username" => username)
    elseif source in ["anilist", "kitsu"]
         r = HTTP.post(
            "$LAYER_3_URL/$(source)_userid",
            encode(Dict("username" => username), :msgpack)...,
            status_exception = false,
        )
        if HTTP.iserror(r)
            return HTTP.Response(r.status, [])
        end
        args = Dict("userid" => decode(r)["userid"])
    end
    r = HTTP.post(
        "$LAYER_3_URL/$(source)_user_parallel",
        encode(args, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r)
        return HTTP.Response(r.status, [])
    end
    d = decode(r)
    for k in keys(d)
        d[k] = normalize(d[k])
    end
    HTTP.Response(200, encode(d, :msgpack)...)
end

Oxygen.@post "/refresh_user_history" function refresh_user_history(r::HTTP.Request)::HTTP.Response
    # unpack
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
    source = data["source"]
    username = data["username"]
    force_refresh = get(data, "force", false)

    # check if the stored version is already up-to-date
    r_read = read_user_history(Dict("source" => source, "username" => username))
    if !HTTP.iserror(r_read)
        d_read = decode(r_read)
        r_fingerprint = HTTP.post(
            "$LAYER_3_URL/$(source)_fingerprint",
            encode(
                Dict(
                    "source" => source,
                    "username" => username,
                    "userid" => d_read["userid"],
                ),
                :msgpack,
            )...,
            status_exception = false,
        )
        if HTTP.iserror(r_fingerprint)
            return HTTP.Response(r_fingerprint.status, [])
        end
        d_fingerprint = decode(r_fingerprint)
        user = MsgPack.unpack(
            CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(d_read["data"])),
        )
        item_df = DataFrames.DataFrame([Dict{String,Any}(x) for x in user["items"]])
        d_read_fingerprint = get_fingerprints(source, item_df)
        if d_read_fingerprint == JSON3.read(d_fingerprint["fingerprint"]) && !force_refresh
            return HTTP.Response(200, encode(Dict("refresh" => false), :msgpack)...)
        end
    end

    # refresh the user's history
    r_fetch = fetch_user_history(Dict("source" => source, "username" => username))
    if HTTP.iserror(r_fetch)
        return HTTP.Response(r_fetch.status, [])
    end
    d_fetch = decode(r_fetch)
    data = merge(
        d_fetch["usermap"],
        Dict(
            "source" => source,
            "data" => CodecZstd.transcode(
                CodecZstd.ZstdCompressor,
                Vector{UInt8}(MsgPack.pack(d_fetch)),
            ),
        ),
    )
    r_write = write_user_history(data)
    if HTTP.iserror(r_write)
        return HTTP.Response(r_write.status, [])
    end
    HTTP.Response(200, encode(Dict("refresh" => true), :msgpack)...)
end

Oxygen.@post "/refresh_user_history" function refresh_user_history(r::HTTP.Request)::HTTP.Response
    # unpack
    if isnothing(LAYER_3_URL)
        return HTTP.Response(403, [])
    end
    data = decode(r)
    source = data["source"]
    username = data["username"]
    force_refresh = get(data, "force", false)

    # check if the stored version is already up-to-date
    r_read = read_user_history(Dict("source" => source, "username" => username))
    if !HTTP.iserror(r_read)
        d_read = decode(r_read)
        r_fingerprint = HTTP.post(
            "$LAYER_3_URL/$(source)_fingerprint",
            encode(
                Dict(
                    "source" => source,
                    "username" => username,
                    "userid" => d_read["userid"],
                ),
                :msgpack,
            )...,
            status_exception = false,
        )
        if HTTP.iserror(r_fingerprint)
            return HTTP.Response(r_fingerprint.status, [])
        end
        d_fingerprint = decode(r_fingerprint)
        user = MsgPack.unpack(
            CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(d_read["data"])),
        )
        item_df = DataFrames.DataFrame([Dict{String,Any}(x) for x in user["items"]])
        d_read_fingerprint = get_fingerprints(source, item_df)
        if d_read_fingerprint == JSON3.read(d_fingerprint["fingerprint"]) && !force_refresh
            return HTTP.Response(200, encode(Dict("refresh" => false), :msgpack)...)
        end
    end

    # refresh the user's history
    r_fetch = fetch_user_history(Dict("source" => source, "username" => username))
    if HTTP.iserror(r_fetch)
        return HTTP.Response(r_fetch.status, [])
    end
    d_fetch = decode(r_fetch)
    data = merge(
        d_fetch["usermap"],
        Dict(
            "source" => source,
            "data" => CodecZstd.transcode(
                CodecZstd.ZstdCompressor,
                Vector{UInt8}(MsgPack.pack(d_fetch)),
            ),
        ),
    )
    r_write = write_user_history(data)
    if HTTP.iserror(r_write)
        return HTTP.Response(r_write.status, [])
    end
    HTTP.Response(200, encode(Dict("refresh" => true), :msgpack)...)
end

Oxygen.@post "/read" function deprecated_read(r::HTTP.Request)::HTTP.Response
    read_user_history(r)
end

Oxygen.@post "/autcomplete" function deprecated_autocomplete(r::HTTP.Request)::HTTP.Response
    read_autocomplete(r)
end

Oxygen.@post "/refresh" function deprecated_refresh(r::HTTP.Request)::HTTP.Response
    refresh_user_history(r)
end

function compile(port::Integer)
    profiles = CSV.read("../../secrets/test.users.csv", DataFrames.DataFrame, stringtype = String)
    @sync for (source, username, userid) in zip(profiles.source, profiles.username, profiles.userid)
        Threads.@spawn begin
            logtag("STARTUP", "/autcomplete")
            HTTP.post(
                "http://localhost:$PORT/autocomplete",
                encode(Dict("source" => source, "prefix" => lowercase(username), "type" => "user"), :msgpack)...,
                status_exception = false,
            )
            logtag("STARTUP", "/read")
            HTTP.post(
                "http://localhost:$PORT/read",
                encode(Dict("source" => source, "username" => username), :msgpack)...,
                status_exception = false,
            )
            logtag("STARTUP", "/refresh")
            HTTP.post(
                "http://localhost:$PORT/refresh",
                encode(Dict("source" => source, "username" => username, "force" => true), :msgpack)...,
                status_exception = false,
            )
        end
    end
end

include("../julia_utils/start_oxygen.jl")

end
