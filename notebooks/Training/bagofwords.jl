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

function parse_args()
    @assert length(ARGS) == 1
    if only(ARGS) == "--pretrain"
        finetune = false
        datadir = "../../data/training"
    elseif only(ARGS) == "--finetune"
        finetune = true
        datadir = "../../data/finetune"
    else
        @assert false
    end
    finetune, datadir
end

const finetune, datadir = parse_args()
const mediums = [0, 1]
const metrics = ["rating", "watch", "plantowatch", "drop"]
const planned_status = 3
const medium_map = Dict(0 => "manga", 1 => "anime")

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

function get_data(data)
    X = Dict()
    Y = Dict()
    W = Dict()
    for m in mediums
        for metric in metrics
            X["$(m)_$(metric)"] = Dict{Int32,Float32}()
            Y["$(m)_$(metric)"] = Dict{Int32,Float32}()
            W["$(m)_$(metric)"] = Dict{Int32,Float32}()
        end
    end
    input_items = data["items"]
    output_items = finetune ? data["test_items"] : data["items"]
    for x in input_items
        m = x["medium"]
        idx = x["matchedid"] + 1
        if x["rating"] > 0
            X["$(m)_rating"][idx] = x["rating"]
        else
            X["$(m)_rating"][idx] = 0
        end
        if x["status"] > planned_status
            X["$(m)_watch"][idx] = 1
        else
            X["$(m)_watch"][idx] = 0
        end
        if x["status"] == planned_status
            X["$(m)_plantowatch"][idx] = 1
        else
            X["$(m)_plantowatch"][idx] = 0
        end
        if x["status"] > 0 && x["status"] < planned_status
            X["$(m)_drop"][idx] = 1
        else
            X["$(m)_drop"][idx] = 0
        end

    end
    for x in output_items
        m = x["medium"]
        idx = x["matchedid"] + 1
        if x["rating"] > 0
            Y["$(m)_rating"][idx] = x["rating"]
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
    ret = Dict()
    for m in mediums
        for metric in metrics
            ret["X_$(m)_$(metric)"] =
                SparseArrays.sparsevec(X["$(m)_$(metric)"], num_items(m))
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

function save_data(datasplit)
    num_shards = finetune ? 1 : 8
    users = sort(Glob.glob("$datadir/users/$datasplit/*/*.msgpack"))
    if datasplit == "test" && !finetune
        users = repeat(users, 5)
    end
    while length(users) % num_shards != 0
        push!(users, rand(users))
    end
    for shard in 1:num_shards
        dest = mkpath("$datadir/bagofwords/$datasplit/$shard")
        files = [x for (i, x) in Iterators.enumerate(users) if (i % num_shards) + 1 == shard]
        files = collect(Iterators.partition(Random.shuffle(files), 65_536))
        @showprogress for p = 1:length(files)
            fns = files[p]
            d = Vector{Any}(undef, length(fns))
            Threads.@threads for i = 1:length(fns)
                data = open(fns[i]) do f
                    MsgPack.unpack(read(f))
                end
                d[i] = get_data(data)
            end
            Random.shuffle!(d)
            h5 = Dict()
            for k in keys(first(d))
                record_sparse_array!(h5, k, sparsecat([x[k] for x in d]))
            end
            HDF5.h5open("$dest/$p.h5", "w") do file
                for (k, v) in h5
                    file[k, blosc = 3] = v
                end
            end
        end
    end
end

function upload()
    if finetune
        return
    end
    template = raw"tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => "$datadir/bagofwords",
        "{OUTPUT}" => "bagofwords",
    )
    run(`sh -c $cmd`)
end

rm("$datadir/bagofwords", recursive = true, force = true)
save_data("test")
save_data("training")
upload()
