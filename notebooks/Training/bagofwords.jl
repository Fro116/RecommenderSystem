import CodecZstd
import CSV
import DataFrames
import Glob
import H5Zblosc
import HDF5
import MsgPack
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random
import SparseArrays

const datadir = "../../data/training"
const envdir = "../../environment"
const mediums = [0, 1]
const metrics = ["rating", "watch", "plantowatch", "drop"]
const planned_status = 3
const medium_map = Dict(0 => "manga", 1 => "anime")
const upload_lock = ReentrantLock()

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame).matchedid) + 1
end

function get_bias(items, m, baselines)
    params = baselines[m]["params"]
    μ_a, λ_u, λ_a, λ_wu, λ_wa = params["λ"]
    λ_u, λ_a = exp.((λ_u, λ_a))
    u = 0
    d = λ_u
    for x in items
        if x["medium"] != m || x["rating"] == 0
            continue
        end
        i = x["matchedid"] + 1
        w = get(params["item_counts"], i, 1)^λ_wa * length(items)^λ_wu
        u += (x["rating"] - params["a"][i]) * w
        d += w
    end
    u / d * only(baselines[m]["weight"])
end

function residualize!(items, biases, baselines)
    for x in items
        if x["rating"] == 0
            x["residual"] = 0
        else
            x["residual"] =
                x["rating"] -
                (biases[x["medium"]] + baselines[x["medium"]]["bias"][x["matchedid"]+1])
        end
    end
end

function get_data(data, baselines, mask_rate, weight_by_user)
    input_items = []
    output_items = []
    for x in data["items"]
        if rand() < mask_rate
            push!(output_items, x)
        else
            push!(input_items, x)
        end
    end
    biases = Dict(m => get_bias(input_items, m, baselines) for m in mediums)
    for x in [input_items, output_items]
        residualize!(x, biases, baselines)
    end
    N = sum(num_items.(mediums))
    X = Dict{Int32,Float32}()
    Y = Dict()
    W = Dict()
    for m in mediums
        for metric in metrics
            Y["$(m)_$(metric)"] = Dict{Int32,Float32}()
            W["$(m)_$(metric)"] = Dict{Int32,Float32}()
        end
    end
    for x in input_items
        idx = x["matchedid"] + 1
        for i = 0:x["medium"]-1
            idx += num_items(i)
        end
        if x["rating"] > 0
            X[idx] = x["residual"]
        else
            X[idx] = 0
        end
        if x["status"] > planned_status
            X[idx+N] = 1
        else
            X[idx+N] = 0
        end
    end
    for x in output_items
        idx = x["matchedid"] + 1
        m = x["medium"]
        if x["rating"] > 0
            Y["$(m)_rating"][idx] = x["residual"]
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
    if weight_by_user
        for m in mediums
            for metric in metrics
                name = "$(m)_$(metric)"
                if isempty(W[name])
                    continue
                end
                t = sum(values(W[name]))
                if t == 0
                    continue
                end
                for k in keys(W[name])
                    W[name][k] /= t
                end
            end
        end
    end
    ret = Dict()
    ret["X"] = SparseArrays.sparsevec(X, sum(num_items.(mediums)) * 2)
    for m in mediums
        for metric in metrics
            ret["Y_$(m)_$(metric)"] =
                SparseArrays.sparsevec(Y["$(m)_$(metric)"], num_items(m))
            ret["W_$(m)_$(metric)"] =
                SparseArrays.sparsevec(W["$(m)_$(metric)"], num_items(m))
        end
    end
    ret
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
end;

function save_epochs(datasplit, epochs, mask_rate, weight_by_user)
    baselines = Dict()
    for m in mediums
        open("$datadir/alphas/baseline.$m.msgpack") do f
            baselines[m] = MsgPack.unpack(read(f))
        end
    end
    rm("$datadir/bagofwords/$datasplit", recursive = true, force = true)
    args = []
    for chunk in collect(Iterators.partition(1:epochs, 8))
        for epoch in chunk
            mkpath("$datadir/bagofwords/$datasplit/$epoch")
        end
        files = collect(
            Iterators.partition(
                Random.shuffle(Glob.glob("$datadir/users/$datasplit/*/*.msgpack")),
                100_000,
            ),
        )
        tasks = [Threads.@spawn upload(datasplit, x) for x in args]
        @showprogress for p = 1:length(files)
            fns = files[p]
            d = Dict(x => Vector{Any}(undef, length(fns)) for x in chunk)
            Threads.@threads for i = 1:length(fns)
                data = open(fns[i]) do f
                    MsgPack.unpack(read(f))
                end
                for epoch in chunk
                    d[epoch][i] = get_data(data, baselines, mask_rate, weight_by_user)
                end
            end
            Threads.@threads for epoch in chunk
                ret = d[epoch]
                Random.shuffle!(ret)
                h5 = Dict()
                for k in keys(first(ret))
                    record_sparse_array!(h5, k, sparsecat([x[k] for x in ret]))
                end
                HDF5.h5open("$datadir/bagofwords/$datasplit/$epoch/$p.h5", "w") do file
                    for (k, v) in h5
                        file[k, blosc = 3] = v
                    end
                end
            end
        end
        fetch.(tasks)
        args = collect(chunk)
    end
    Threads.@threads for x in args
        upload(datasplit, x)
    end
end

function upload(datasplit, epoch)
    lock(upload_lock) do
        template = read("$envdir/database/upload.txt", String)
        cmd = replace(
            template,
            "{INPUT}" => "$datadir/bagofwords/$datasplit/$epoch",
            "{OUTPUT}" => "bagofwords/$datasplit/$epoch",
        )
        run(`sh -c $cmd`)
        rm("$datadir/bagofwords/$datasplit/$epoch", recursive = true, force = true)
    end
end

save_epochs("test", 1, 0.1, true)
save_epochs("training", 64, 0.25, false)
