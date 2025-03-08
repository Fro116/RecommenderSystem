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
const metrics = ["rating", "watch", "plantowatch", "drop"]
const planned_status = 3
const medium_map = Dict(0 => "manga", 1 => "anime")
const min_ts = Dates.datetime2unix(Dates.DateTime("20000101", Dates.dateformat"yyyymmdd"))
const max_ts = Dates.datetime2unix(
    Dates.DateTime(read("$datadir/latest", String), Dates.dateformat"yyyymmdd"),
)
const max_seq_len = 1024
const batch_size = 256 * max_seq_len

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame).matchedid) + 1
end

@memoize function get_baselines()
    baselines = Dict()
    for m in [0, 1]
        b = open("$datadir/baseline.$m.msgpack") do f
            MsgPack.unpack(read(f))
        end
        d = Dict()
        d["params"] = convert.(Float32, b["params"]["λ"])
        d["weight"] = convert(Float32, only(b["weight"]))
        d["a"] = convert.(Float32, b["params"]["a"])
        item_counts = b["params"]["item_counts"]
        item_counts = Int32[get(item_counts, x, 1) for x = 0:length(b["params"]["a"])-1]
        d["item_counts"] = item_counts
        baselines[m] = d
    end
    baselines
end

function get_user_bias(data)
    baselines = get_baselines()
    biases = Dict()
    for m in [0, 1]
        _, λ_u, _, λ_wu, λ_wa = baselines[m]["params"]
        numer = 0
        denom = exp(λ_u)
        ucount = length([x for x in data["items"] if x["rating"] > 0 && x["medium"] == m])
        for x in data["items"]
            if x["medium"] != m || x["rating"] == 0
                continue
            end
            acount = baselines[m]["item_counts"][x["matchedid"]+1]
            w = ucount^λ_wu * acount^λ_wa
            numer += (x["rating"] - baselines[m]["a"][x["matchedid"]+1]) * w
            denom += w
        end
        biases[m] = numer / denom
    end
    biases
end

function get_data(data, userid)
    biases = get_user_bias(data)
    reserved_vals = 2
    cls_val = -1
    mask_val = -2
    d = Dict{String,Any}(
        "status" => zeros(Int32, max_seq_len),
        "rating" => zeros(Float32, max_seq_len),
        "progress" => zeros(Float32, max_seq_len),
        "0_matchedid" => zeros(Int32, max_seq_len),
        "0_distinctid" => zeros(Int32, max_seq_len),
        "1_matchedid" => zeros(Int32, max_seq_len),
        "1_distinctid" => zeros(Int32, max_seq_len),
        "updated_at" => zeros(Float32, max_seq_len),
        "source" => zeros(Int32, max_seq_len),
        "position" => zeros(Int32, max_seq_len),
    )
    input_fields = collect(keys(d))
    d["userid"] = zeros(Int32, max_seq_len)
    d["mask_index"] = zeros(Int32, max_seq_len)
    for k in input_fields
        d[k][1] = cls_val
    end
    d["userid"][1] = userid
    d["rating"][1] = 0
    skipped = 0
    items = data["items"]
    while length(items) > max_seq_len - 2
        items = items[2:end]
        skipped += 1
    end
    i = 1
    for x in items
        i += 1
        m = x["medium"]
        n = 1 - x["medium"]
        d["status"][i] = x["status"]
        if x["rating"] > 0
            bl = get_baselines()[m]
            rating_pred = (biases[m] + bl["a"][x["matchedid"]+1]) * bl["weight"]
            d["rating"][i] = x["rating"] - rating_pred
        else
            d["rating"][i] = 0
        end
        d["position"][i] = (i + skipped) % (max_seq_len - reserved_vals)
        d["progress"][i] = x["progress"]
        d["$(m)_matchedid"][i] = x["matchedid"]
        d["$(m)_distinctid"][i] = x["distinctid"]
        d["$(n)_matchedid"][i] = cls_val
        d["$(n)_distinctid"][i] = cls_val
        d["updated_at"][i] = clamp((x["updated_at"] - min_ts) / (max_ts - min_ts), 0, 1)
        d["source"][i] = data["user"]["source"]
        d["userid"][i] = userid
        d["mask_index"][i] = 0
    end
    i += 1
    for k in input_fields
        d[k][i] = mask_val
    end    
    d["userid"][i] = userid
    d["updated_at"][i] = 1
    d["position"][i] = i % (max_seq_len - reserved_vals)
    d["mask_index"][i] = 1
    d["rating"][i] = 0
    Y = Dict()
    W = Dict()
    for m in mediums
        for metric in metrics
            Y["$(m)_$(metric)"] = Dict{Int32,Float32}()
            W["$(m)_$(metric)"] = Dict{Int32,Float32}()
        end
    end
    for x in data["test_items"]
        m = x["medium"]
        idx = x["matchedid"] + 1
        if x["rating"] > 0
            bl = get_baselines()[m]
            rating_pred = (biases[m] + bl["a"][idx]) * bl["weight"]
            Y["$(m)_rating"][idx] = x["rating"] - rating_pred
            W["$(m)_rating"][idx] = 1
        else
            Y["$(m)_rating"][idx] = 0
            W["$(m)_rating"][idx] = 0
        end
        if x["status"] > planned_status
            Y["$(m)_watch"][idx] = 1
            W["$(m)_watch"][idx] = 1
        else
            Y["$(m)_watch"][idx] = 0
            W["$(m)_watch"][idx] = 0
        end
        if x["status"] == planned_status
            Y["$(m)_plantowatch"][idx] = 1
            W["$(m)_plantowatch"][idx] = 1
        else
            Y["$(m)_plantowatch"][idx] = 0
            W["$(m)_plantowatch"][idx] = 0
        end
        if x["status"] > 0 && x["status"] < planned_status
            Y["$(m)_drop"][idx] = 1
            W["$(m)_drop"][idx] = 1
        else
            Y["$(m)_drop"][idx] = 0
            W["$(m)_drop"][idx] = 1
        end
    end
    for m in mediums
        for metric in metrics
            N = num_items(m) + reserved_vals
            d["$m.$metric.label"] = SparseArrays.sparsevec(Y["$(m)_$(metric)"], N)
            d["$m.$metric.weight"] = SparseArrays.sparsevec(W["$(m)_$(metric)"], N)
            wsum = sum(d["$m.$metric.weight"])
            if wsum > 0
                d["$m.$metric.weight"] /= wsum
            end
        end
    end
    d
end

function sparsecat(xs::Vector{SparseArrays.SparseVector{Tv,Ti}}) where {Tv,Ti}
    N = sum(SparseArrays.nnz.(xs))
    I = Vector{Int32}(undef, N)
    J = Vector{Int32}(undef, N)
    V = Vector{eltype(first(xs))}(undef, N)
    idx = 1
    for (j, x) in Iterators.enumerate(xs)
        i, v = SparseArrays.findnz(x)
        for idy = 1:length(i)
            I[idx] = i[idy]
            J[idx] = j
            V[idx] = v[idy]
            idx += 1
        end
    end
    SparseArrays.sparse(I, J, V, only(size(first(xs))), length(xs))
end

function record_sparse_array!(d, name, x)
    i, j, v = SparseArrays.findnz(x)
    d[name*"_i"] = i
    d[name*"_j"] = j
    d[name*"_v"] = v
    d[name*"_size"] = collect(size(x))
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
            if first(d)[k] isa SparseArrays.SparseVector
                record_sparse_array!(h5, k, sparsecat([x[k] for x in d]))
            else
                h5[k] = reduce(hcat, [x[k] for x in d])
            end
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
