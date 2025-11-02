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
import SparseArrays

const datadir = "../../data/finetune"
const mediums = [0, 1]
const metrics = ["watch", "rating", "status"]
const planned_status = 5
const medium_map = Dict(0 => "manga", 1 => "anime")
const max_ts = Dates.datetime2unix(
    Dates.DateTime(read("$datadir/finetune_tag", String), Dates.dateformat"yyyymmdd"),
) + 86400
const max_seq_len = 1024
const num_gpus = 8

include("../Training/history_tools.jl")

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

@memoize function get_baselines()
    baselines = Dict()
    for metric in ["rating"]
        baselines[metric] = Dict()
        for m in [0, 1]
            b = open("$datadir/baseline.$metric.$m.msgpack") do f
                MsgPack.unpack(read(f))
            end
            d = Dict()
            d["params"] = convert.(Float32, b["params"]["λ"])
            d["weight"] = convert(Float32, only(b["weight"]))
            d["a"] = convert.(Float32, b["params"]["a"])
            item_counts = b["params"]["item_counts"]
            item_counts = Int32[get(item_counts, x, 1) for x = 0:length(b["params"]["a"])-1]
            d["item_counts"] = item_counts
            baselines[metric][m] = d
        end
    end
    baselines
end

function get_data(data, userid)
    tokenize!(data)
    project!(data)
    num_test_items = 1
    @assert length(data["test_items"]) <= num_test_items
    N = max_seq_len # TODO extend seq_len
    d = Dict(
        # prompt features
        "userid" => zeros(Int32, N),
        "time" => zeros(Float64, N),
        "gender" => zeros(Int32, N),
        "source" => zeros(Int32, N),
        # item features
        "matchedid" => zeros(Int32, N),
        # action features
        "status" => zeros(Int32, N),
        "rating" => zeros(Float32, N),
        "progress" => zeros(Float32, N),
    )
    # targets
    for m in [0, 1]
        for metric in metrics
            d["$m.$metric.label"] = zeros(Float32, N)
            d["$m.$metric.weight"] = zeros(Float32, N)
            d["$m.$metric.position"] = zeros(Int32, N)
        end
    end
    items = data["items"]
    if length(items) > max_seq_len - num_test_items
        items = items[length(items) - (max_seq_len-num_test_items) + 1:end]
    end
    u = data["user"]
    i = 1
    for (item_source, istest) in [(items, false), (data["test_items"], true)]
        for x in item_source
            m = x["medium"]
            # prompt features
            d["userid"][i] = userid
            d["time"][i] = x["history_max_ts"]
            d["gender"][i] = isnothing(u["gender"]) ? 0 : u["gender"] + 1
            d["source"][i] = u["source"]
            # item features
            d["matchedid"][i] = x["matchedid"] + ((m == 1) ? num_items(0) : 0)
            # action features
            d["status"][i] = x["status"]
            d["rating"][i] = x["rating"]
            d["progress"][i] = x["progress"]
            # targets
            if istest
                inferred_watch = x["status"] == 0 && isnothing(x["history_status"])
                new_watch = (x["status"] > planned_status) && (isnothing(x["history_status"]) || 0 < x["history_status"] <= planned_status)
                if inferred_watch || new_watch
                    d["$m.watch.label"][i] = 1
                    d["$m.watch.weight"][i] = 1
                    d["$m.watch.position"][i] = x["matchedid"]
                end
                if (x["rating"] > 0) && (x["rating"] != x["history_rating"])
                    d["$m.rating.label"][i] = x["rating"]
                    d["$m.rating.weight"][i] = 1
                    d["$m.rating.position"][i] = x["matchedid"]
                end
                if (x["status"] > 0) && (x["status"] != x["history_status"])
                    d["$m.status.label"][i] = x["status"]
                    d["$m.status.weight"][i] = 1
                    d["$m.status.position"][i] = x["matchedid"]
                end
            end
            i += 1
        end
    end
    for m in mediums
        for metric in metrics
            wsum = sum(d["$m.$metric.weight"])
            if wsum > 0
                d["$m.$metric.weight"] /= wsum
            end
        end
    end
    d
end

function save_data(datasplit)
    num_shards = num_gpus
    users = sort(Glob.glob("$datadir/users/$datasplit/*/*.msgpack"))
    while length(users) % num_shards != 0
        push!(users, rand(users))
    end
    Random.shuffle!(users)
    for shard = 1:num_shards
        dest = mkpath("$datadir/transformer/$datasplit/$shard")
        files = [x for (i, x) in Iterators.enumerate(users) if (i % num_shards) + 1 == shard]
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
            Random.shuffle!(d)
            h5 = Dict()
            for k in keys(first(d))
                h5[k] = reduce(hcat, [x[k] for x in d])
            end
            HDF5.h5open("$dest/$p.h5", "w") do file
                for (k, v) in h5
                    file[k, blosc = 3] = v
                end
            end
        end
    end
end

rm("$datadir/transformer", recursive = true, force = true)
save_data("test")
save_data("training")
