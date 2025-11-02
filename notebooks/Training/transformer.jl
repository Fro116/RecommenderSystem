import CSV
import DataFrames
import Dates
import Glob
import JLD2
import JSON3
import H5Zblosc
import HDF5
import MsgPack
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random
import SparseArrays

const datadir = "../../data/training"
const mediums = [0, 1]
const metrics = ["watch", "rating", "status"]
const planned_status = 5
const medium_map = Dict(0 => "manga", 1 => "anime")
const batch_size = 128 * 1024 # local_batch_size * max_sequence_length
const num_gpus = 8
const min_ts = Dates.datetime2unix(Dates.DateTime("2000-01-01"))
const max_ts = Dates.datetime2unix(
    Dates.DateTime(read("$datadir/list_tag", String), Dates.dateformat"yyyymmdd"),
)

include("../julia_utils/stdout.jl")
include("history_tools.jl")

get_transdir(mini) = mini ? "transformer_mini" : "transformer"

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

function optdate(x)
    if ismissing(x) || isnothing(x) || isempty(x)
        return 0, 0
    end
    try
        ts = Dates.datetime2unix(Dates.DateTime(Dates.Date(x)))
        y = (ts - min_ts) / (max_ts - min_ts)
        y = clamp(y, -5, 5)
        return 1, y
    catch
        logerror("could not parse date $x")
        fields = split(x, "-")
        if length(fields) > 1
            return optdate(join(fields[1:end-1], "-"))
        end
    end
    0, 0
end

function save_media_embeddings()
    d = Dict()
    W = zeros(Float32, 3072 + 4, sum(num_items.([0, 1])))
    for medium in [0, 1]
        m = Dict(0 => "manga", 1 => "anime")[medium]
        data = JSON3.read("$datadir/$m.json")
        for x in data
            embs = sum(values(x[:embedding])) ./ length(x[:embedding])
            has_sd, sd = optdate(x[:metadata][:dates][:startdate])
            has_ed, ed = optdate(x[:metadata][:dates][:enddate])
            idx = x[:matchedid]+1 + ((medium == 1) ? num_items(0) : 0)
            W[:, idx] = vcat(embs, [has_sd, sd, has_ed, ed])
        end
        d["metadata"] = W
    end
    HDF5.h5open("$datadir/media_embeddings.h5", "w") do file
        for (k, v) in d
            file[k, blosc = 3] = v
        end
    end
end

function get_data(data, userid)
    tokenize!(data)
    project!(data)
    u = data["user"]
    N = length(data["items"])
    d = Dict(
        # prompt features
        "userid" => Vector{Int32}(undef, N),
        "time" => Vector{Float64}(undef, N),
        "gender" => Vector{Int32}(undef, N),
        "source" => Vector{Int32}(undef, N),
        # item features
        "matchedid" => Vector{Int32}(undef, N),
        # action features
        "status" => Vector{Int32}(undef, N),
        "rating" => Vector{Float32}(undef, N),
        "progress" => Vector{Float32}(undef, N),
    )
    # targets
    for m in [0, 1]
        for metric in metrics
            d["$m.$metric.label"] = zeros(Float32, N)
            d["$m.$metric.weight"] = zeros(Float32, N)
            d["$m.$metric.position"] = zeros(Int32, N)
        end
    end
    for (i, x) in Iterators.enumerate(data["items"])
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
        inferred_watch = x["status"] == 0 && isnothing(x["history_status"])
        new_watch =
            (x["status"] > planned_status) &&
            (isnothing(x["history_status"]) || 0 < x["history_status"] <= planned_status)
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

function get_num_tokens(datasplit, shard, transdir::String)
    tokens = 0
    fns = Glob.glob("$datadir/$transdir/$datasplit/$shard/*.h5")
    for fn in fns
        HDF5.h5open(fn) do file
            tokens += length(file["userid"])
        end
    end
    tokens
end

function pad_splits(datasplit, num_shards, transdir::String)
    max_num_tokens = maximum(get_num_tokens(datasplit, x, transdir) for x = 1:num_shards)
    for x = 1:num_shards
        num_padding = max_num_tokens - get_num_tokens(datasplit, x, transdir)
        if num_padding == 0
            continue
        end
        logtag(
            "TRANSFORMER",
            "padding split $x with $num_padding tokens to reach $max_num_tokens tokens",
        )
        infn = "$datadir/$transdir/$datasplit/$x/1.h5"
        outfn = "$datadir/$transdir/$datasplit/$x/pad.h5"
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

function save_data(datasplit, mini::Bool)
    transdir = get_transdir(mini)
    num_shards = num_gpus
    users = sort(Glob.glob("$datadir/users/$datasplit/*/*.msgpack"))
    if mini
        users = [x for x in users if rand() < 0.5]
    end
    while length(users) % num_shards != 0
        push!(users, rand(users))
    end
    Random.shuffle!(users)
    for shard = 1:num_shards
        dest = mkpath("$datadir/$transdir/$datasplit/$shard")
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
    pad_splits(datasplit, num_shards, transdir)
    total_tokens = sum(get_num_tokens.(datasplit, 1:num_shards, transdir))
    open("$datadir/$transdir/$datasplit/num_tokens.txt", "w") do f
        write(f, "$total_tokens")
    end
end

function upload()
    template =
        raw"tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    for fn in [get_transdir.([true, false]); ["media_embeddings.h5"]]
        cmd =
            replace(template, "{INPUT}" => "$datadir/$fn", "{OUTPUT}" => fn)
        run(`sh -c $cmd`)
    end
end

save_media_embeddings()
for mini in [true, false]
    rm("$datadir/$(get_transdir(mini))", recursive = true, force = true)
    save_data("test", mini)
    save_data("training", mini)
end
upload()
