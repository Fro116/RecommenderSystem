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
    project!(data)
    num_test_items = 5 # up to 5 test items
    @assert length(data["test_items"]) <= num_test_items
    N = max_seq_len # TODO extend seq_len
    d = Dict(
        # prompt features
        "userid" => zeros(Int32, N),
        "time" => zeros(Float64, N),
        "input_pos" => zeros(Int32, N),
        # item features
        "0_matchedid" => zeros(Int32, N),
        "0_distinctid" => zeros(Int32, N),
        "1_matchedid" => zeros(Int32, N),
        "1_distinctid" => zeros(Int32, N),
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
        end
    end
    items = data["items"]
    input_pos = 0
    if length(items) > max_seq_len - num_test_items
        input_pos = length(items) - (max_seq_len-num_test_items)
        items = items[input_pos + 1:end]
    end
    i = 1
    for (item_source, istest) in [(items, false), (data["test_items"], true)]
        for x in item_source
            m = x["medium"]
            n = 1 - x["medium"]
            # prompt features
            d["userid"][i] = userid
            d["time"][i] = x["history_max_ts"]
            d["input_pos"][i] = input_pos
            # item features
            d["$(m)_matchedid"][i] = x["matchedid"]
            d["$(m)_distinctid"][i] = x["distinctid"]
            d["$(n)_matchedid"][i] = -1
            d["$(n)_distinctid"][i] = -1
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
                end
                if (x["rating"] > 0) && (x["rating"] != x["history_rating"])
                    d["$m.rating.label"][i] = x["rating"]
                    d["$m.rating.weight"][i] = 1
                end
                if (x["status"] > 0) && (x["status"] != x["history_status"])
                    d["$m.status.label"][i] = x["status"]
                    d["$m.status.weight"][i] = 1
                end
            end
            i += 1
            input_pos += 1
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
    users = sort(Glob.glob("$datadir/users/$datasplit/*/*.msgpack"))
    dest = mkpath("$datadir/transformer/$datasplit/1")
    files = collect(Iterators.partition(Random.shuffle(users), 65_536))
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

rm("$datadir/transformer", recursive = true, force = true)
save_data("test")
save_data("training")
