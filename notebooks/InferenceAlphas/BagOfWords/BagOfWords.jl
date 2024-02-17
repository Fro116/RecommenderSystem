import NBInclude: @nbinclude
import Memoize: @memoize

if !@isdefined BAGOFWORDS_IFNDEF
    BAGOFWORDS_IFNDEF = true

    source_name = "BagOfWords"
    import H5Zblosc
    import HDF5
    import JSON
    import SparseArrays: AbstractSparseArray, sparse
    @nbinclude("../../TrainingAlphas/Alpha.ipynb")

    @memoize function get_rating_beta(name)
        params = read_params(name, false)
        params["β"]
    end

    function get_inputs(medium::String, metric::String)
        split = "rec_training"
        fields = [:userid, :itemid, :metric]
        if metric == "rating"
            alpha = "$medium/Baseline/rating"
            β = get_rating_beta(alpha)
            df = get_split(split, metric, medium, fields, alpha)
            df.metric .= df.metric - df.alpha .* β
        else
            df = get_split(split, metric, medium, fields)
        end
        sparse(df, medium)
    end

    function get_epoch_inputs()
        inputs = [
            get_inputs(medium, metric) for metric in ["rating", "watch"] for
            medium in ALL_MEDIUMS
        ]
        vcat(inputs...)
    end

    function save_features(outdir)
        mkpath(outdir)
        filename = joinpath(outdir, "inference.h5")
        d = Dict{String,Any}()
        X = get_epoch_inputs()
        record_sparse_array!(d, "inputs", X)
        d["epoch_size"] = 1
        d["users"] = [0]
        HDF5.h5open(filename, "w") do file
            for (k, v) in d
                file[k, blosc = 1] = v
            end
        end
    end

    function record_sparse_array!(d::Dict, name::String, x::AbstractSparseArray)
        i, j, v = SparseArrays.findnz(x)
        d[name*"_i"] = i
        d[name*"_j"] = j
        d[name*"_v"] = v
        d[name*"_size"] = [size(x)[1], size(x)[2]]
    end

    function compute_alpha(source, username, medium, metric, version)
        outdir = joinpath(
            get_data_path("recommendations/$(get_rec_usertag())"),
            "alphas",
            medium,
            "BagOfWords",
            version,
            metric,
        )
        save_features(outdir)
        cmd = (
            "python3 BagOfWords.py --source $source --username $username --medium $medium" *
            " --metric $metric --version $version"
        )
        run(Cmd(convert(Vector{String}, split(cmd))))
        file = HDF5.h5open(joinpath(outdir, "predictions.h5"), "r")
        e = vec(read(file["predictions"]))
        seen = get_raw_split("rec_training", medium, [:itemid], nothing).itemid
        if metric in ["watch", "plantowatch"]
            e[seen.+1] .= 0 # zero out watched items
            e = e ./ sum(e)
        elseif metric in ["rating", "drop"]
            nothing
        else
            @assert false
        end
        close(file)
        model(userids, itemids) = [e[x+1] for x in itemids]
        write_alpha(model, medium, "$medium/BagOfWords/$version/$metric", REC_SPLITS)
    end
end

username = ARGS[1]
source = ARGS[2]
version = "v1"
for medium in ALL_MEDIUMS
    for metric in ALL_METRICS
        compute_alpha(source, username, medium, metric, version)
    end
end