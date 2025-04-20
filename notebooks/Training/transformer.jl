import CSV
import DataFrames
import Dates
import Glob
import H5Zblosc
import HDF5
import MsgPack
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random

const datadir = "../../data/training"
const mediums = [0, 1]
const metrics = ["watch", "rating", "status"]
const planned_status = 4
const medium_map = Dict(0 => "manga", 1 => "anime")
const min_ts = Dates.datetime2unix(Dates.DateTime("20000101", Dates.dateformat"yyyymmdd"))
const max_ts = Dates.datetime2unix(
    Dates.DateTime(read("$datadir/latest", String), Dates.dateformat"yyyymmdd"),
)
const max_seq_len = 1024
const batch_size = 64 * max_seq_len

include("../julia_utils/stdout.jl")

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

function get_data(data, userid)
    cls_val = -1
    N = length(data["items"]) + 1
    d = Dict(
        "status" => Vector{Int32}(undef, N),
        "rating" => Vector{Float32}(undef, N),
        "progress" => Vector{Float32}(undef, N),
        "0_matchedid" => Vector{Int32}(undef, N),
        "0_distinctid" => Vector{Int32}(undef, N),
        "1_matchedid" => Vector{Int32}(undef, N),
        "1_distinctid" => Vector{Int32}(undef, N),
        "time" => zeros(Float64, N),
        "delta_time" => zeros(Float64, N),
    )
    for k in keys(d)
        d[k][1] = cls_val
    end
    d["userid"] = fill(Int32(userid), N)
    for m in [0, 1]
        for metric in metrics
            d["$m.$metric.label"] = zeros(Float32, N)
            d["$m.$metric.weight"] = zeros(Float32, N)
        end
    end
    last_ts = min_ts
    i = 1
    for x in data["items"]
        i += 1
        d["status"][i] = x["status"]
        d["rating"][i] = x["rating"]
        d["progress"][i] = x["progress"]
        m = x["medium"]
        n = 1 - x["medium"]
        d["$(m)_matchedid"][i] = x["matchedid"]
        d["$(m)_distinctid"][i] = x["distinctid"]
        d["$(n)_matchedid"][i] = cls_val
        d["$(n)_distinctid"][i] = cls_val
        if x["updated_at"] >= min_ts && x["updated_at"] <= max_ts
            d["time"][i] = x["updated_at"]
            d["delta_time"][i-1] = x["updated_at"] - last_ts
            last_ts = x["updated_at"]
        end
        if (x["status"] == 0 || x["status"] >= planned_status) && (x["rating"] == 0 || x["rating"] >= 5)
            d["$m.watch.label"][i] = 1
            d["$m.watch.weight"][i] = 1
        end
        if x["rating"] > 0
            d["$m.rating.label"][i] = x["rating"]
            d["$m.rating.weight"][i] = 1
        end
        if x["status"] > 0
            d["$m.status.label"][i] = x["status"]
            d["$m.status.weight"][i] = 1
        end
    end
    d
end

function concat(d)
    N = sum(length(x["userid"]) for x in d)
    if N % batch_size != 0
        N += batch_size - (N % batch_size)
    end
    r = Dict()
    for (k, v) in d[1]
        r[k] = zeros(eltype(v), N)
    end
    Threads.@threads for k in collect(keys(r))
        i = 1
        for x in d
            for v in x[k]
                r[k][i] = v
                i += 1
            end
        end
    end
    r
end

function get_num_tokens(datasplit, shard)
    tokens = 0
    fns = Glob.glob("$datadir/transformer/$datasplit/$shard/*.h5")
    for fn in fns
        HDF5.h5open(fn) do file
            tokens += length(file["userid"])
        end
    end
    tokens
end

function pad_splits(datasplit, num_shards)
    max_num_tokens = maximum(get_num_tokens(datasplit, x) for x = 1:num_shards)
    for x = 1:num_shards
        num_padding = max_num_tokens - get_num_tokens(datasplit, x)
        if num_padding == 0
            continue
        end
        logtag("TRANSFORMER", "padding split $x with $num_padding tokens")
        infn = "$datadir/transformer/$datasplit/$x/1.h5"
        outfn = "$datadir/transformer/$datasplit/$x/pad.h5"
        HDF5.h5open(outfn, "w") do outfile
            HDF5.h5open(infn) do file
                for k in keys(file)
                    n = min(num_padding, length(file[k]))
                    x = zeros(eltype(file[k]), num_padding)
                    x[1:n] = file[k][1:n]
                    outfile[k, blosc = 3] = x
                end
            end
        end
    end
end

function save_data(datasplit)
    num_shards = 8
    users = sort(Glob.glob("$datadir/users/$datasplit/*/*.msgpack"))
    while length(users) % num_shards != 0
        push!(users, rand(users))
    end
    Random.shuffle!(users)
    for shard = 1:num_shards
        dest = mkpath("$datadir/transformer/$datasplit/$shard")
        files =
            [x for (i, x) in Iterators.enumerate(users) if (i % num_shards) + 1 == shard]
        files = collect(Iterators.partition(Random.shuffle(files), 65_536))
        @showprogress for p = 1:length(files)
            fns = files[p]
            d = Vector{Any}(undef, length(fns))
            Threads.@threads for i = 1:length(fns)
                data = open(fns[i]) do f
                    MsgPack.unpack(read(f))
                end
                d[i] = get_data(data, i)
            end
            h5 = concat(d)
            HDF5.h5open("$dest/$p.h5", "w") do file
                for (k, v) in h5
                    file[k, blosc = 3] = v
                end
            end
        end
    end
    pad_splits(datasplit, num_shards)
    total_tokens = sum(get_num_tokens.(datasplit, 1:num_shards))
    open("$datadir/transformer/$datasplit/num_tokens.txt", "w") do f
        write(f, "$total_tokens")
    end
end

function upload()
    template = raw"tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => "$datadir/transformer",
        "{OUTPUT}" => "transformer",
    )
    run(`sh -c $cmd`)
end

function set_random_seed()
    tag = read("$datadir/latest", String)
    Random.seed!(tag)
end

rm("$datadir/transformer", recursive = true, force = true)
set_random_seed()
save_data("test")
save_data("training")
upload()
