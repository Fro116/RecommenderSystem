import CodecZstd
import Dates
import DataFrames
import HTTP
import MsgPack
import Oxygen

include("notebooks/Preprocess/ImportLists/import_lists.jl")

pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
unpack(d::Vector{UInt8}) =
    MsgPack.unpack(CodecZstd.transcode(CodecZstd.ZstdDecompressor, d))

function msgpack(d::Dict)::HTTP.Response
    body = pack(d)
    headers = Dict(
        "Content-Type" => "application/msgpack",
    )
    HTTP.Response(200, headers, body)
end

READY::Bool = false

Oxygen.@get "/ready" function ready(req::HTTP.Request)
    if !READY
        return HTTP.Response(503, [])
    else
        return HTTP.Response(200, [])
    end
end

Oxygen.@get "/terminate" terminate(req::HTTP.Request) = Oxygen.terminate()

struct UnitaryDict
    value::Any
end
Base.get(x::UnitaryDict, k) = x.value;
Base.get(x::UnitaryDict, k, default) = x.value;

Oxygen.@post "/query" function query(req::HTTP.Request)
    if !READY && !HTTP.Messages.hasheader(req, "Startup")
        return HTTP.Response(503, [])
    end
    qp = Oxygen.queryparams(req)
    medium, source = qp["medium"], qp["source"]
    df = DataFrames.DataFrame(unpack(req.body))
    max_valid_ts = Int(round(Dates.datetime2unix(Dates.now() + Dates.Day(1))))
    df = import_list(medium, source, UnitaryDict(1), max_valid_ts, df)
    msgpack(Dict(k => getfield(df, k) for k in fieldnames(typeof(df))))
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

function test_query_inputs(source, medium)
    Dict(
        ("mal", "manga") => Dict(
            "uid" => ["598"],
            "status" => ["dropped"],
            "score" => ["6"],
            "progress" => ["0"],
            "progress_volumes" => ["0"],
            "started_at" => [""],
            "completed_at" => [""],
            "priority" => [""],
            "repeat" => ["False"],
            "repeat_count" => [""],
            "repeat_value" => [""],
            "tags" => [""],
            "notes" => [""],
            "updated_at" => ["1478489023"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("mal", "anime") => Dict(
            "uid" => ["32995"],
            "status" => ["watching"],
            "score" => ["8"],
            "progress" => ["0"],
            "progress_volumes" => [""],
            "started_at" => [""],
            "completed_at" => [""],
            "priority" => [""],
            "repeat" => ["False"],
            "repeat_count" => [""],
            "repeat_value" => [""],
            "tags" => [""],
            "notes" => [""],
            "updated_at" => ["1478489355"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("anilist", "manga") => Dict(
            "anilistid" => ["151058"],
            "malid" => ["152435"],
            "score" => ["0.0"],
            "status" => ["PLANNING"],
            "progress" => ["0"],
            "progress_volumes" => ["0"],
            "repeat" => ["0"],
            "priority" => ["0"],
            "notes" => [""],
            "started_at" => ["--"],
            "completed_at" => ["--"],
            "updated_at" => ["1665767272"],
            "created_at" => ["1665767272"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("anilist", "anime") => Dict(
            "anilistid" => ["108725"],
            "malid" => ["39617"],
            "score" => ["0"],
            "status" => ["COMPLETED"],
            "progress" => ["11"],
            "progress_volumes" => [""],
            "repeat" => ["1"],
            "priority" => ["0"],
            "notes" => [""],
            "started_at" => ["--"],
            "completed_at" => ["2024-9-1"],
            "updated_at" => ["1725167197"],
            "created_at" => ["1725167191"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("kitsu", "manga") => Dict(
            "kitsuid" => ["20055"],
            "malid" => ["40225"],
            "score" => ["0.0"],
            "status" => ["planned"],
            "progress" => ["0"],
            "volumes_owned" => ["0"],
            "repeat" => ["False"],
            "repeat_count" => ["0"],
            "notes" => [""],
            "private" => ["False"],
            "reaction_skipped" => ["unskipped"],
            "progressed_at" => ["1707670049"],
            "updated_at" => ["1707670049"],
            "created_at" => ["1707670049"],
            "started_at" => ["0"],
            "finished_at" => ["0"],
            "usertag" => ["test"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("kitsu", "anime") => Dict(
            "kitsuid" => ["1555"],
            "malid" => ["1735"],
            "score" => ["0.0"],
            "status" => ["planned"],
            "progress" => ["0"],
            "volumes_owned" => ["0"],
            "repeat" => ["False"],
            "repeat_count" => ["0"],
            "notes" => [""],
            "private" => ["False"],
            "reaction_skipped" => ["unskipped"],
            "progressed_at" => ["1707669989"],
            "updated_at" => ["1707669989"],
            "created_at" => ["1707669989"],
            "started_at" => ["0"],
            "finished_at" => ["0"],
            "usertag" => ["test"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("animeplanet", "manga") => Dict(
            "title" => ["Yuuna and the Haunted Hot Springs"],
            "url" => ["yuuna-and-the-haunted-hot-springs"],
            "score" => ["7.0"],
            "status" => ["1"],
            "progress" => ["0"],
            "updated_at" => ["1705052316"],
            "item_order" => ["0"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
        ("animeplanet", "anime") => Dict(
            "title" => ["That Time I Got Reincarnated as a Slime"],
            "url" => ["that-time-i-got-reincarnated-as-a-slime"],
            "score" => ["10.0"],
            "status" => ["1"],
            "progress" => ["0"],
            "updated_at" => ["1705052300"],
            "item_order" => ["0"],
            "api_version" => ["4.1.0"],
            "username" => ["test"],
        ),
    )[(source, medium)]
end

function test_query(port)
    pack(d::Dict) = CodecZstd.transcode(CodecZstd.ZstdCompressor, MsgPack.pack(d))
    for source in ["mal", "anilist", "kitsu", "animeplanet"]
        for medium in ["manga", "anime"]
            url = "http://localhost:$port/query?source=$source&medium=$medium"
            body = pack(test_query_inputs(source, medium))
            headers = Dict(
                "Content-Type" => "application/msgpack",
                "Startup" => true,
            )
            HTTP.post(url, headers, body)
        end
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
