import NBInclude: @nbinclude
import Memoize: @memoize

if !@isdefined IFNDEF
    IFNDEF = true

    import H5Zblosc
    import HDF5
    import JSON
    import SparseArrays: AbstractSparseArray, sparse
    @nbinclude("../TrainingAlphas/Alpha.ipynb")

    @memoize function get_rating_beta(name)
        params = read_params(name, false)
        params["β"]
    end

    function get_inputs(medium::String, metric::String, username::String, source::String)
        split = "rec_training"
        fields = [:userid, :itemid, :metric]
        if metric == "rating"
            alpha = "$medium/Baseline/rating"
            β = get_rating_beta(alpha)
            df = get_split(split, metric, medium, fields, alpha, username, source)
            df.metric .= df.metric - df.alpha .* β
        else
            df = get_split(split, metric, medium, fields, nothing, username, source)
        end
        sparse(df, medium)
    end

    function get_epoch_inputs(username::String, source::String)
        GC.enable(false)
        inputs = [
            get_inputs(medium, metric, username, source) for
            metric in ["rating", "watch"] for medium in ALL_MEDIUMS
        ]
        X = vcat(inputs...)
        GC.enable(true)
        X
    end

    function save_features(outdir::String, username::String, source::String)
        mkpath(outdir)
        filename = joinpath(outdir, "inference.h5")
        d = Dict{String,Any}()
        X = get_epoch_inputs(username, source)
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

    function compute_alpha(username, source, medium, metric, version)
        cmd = (
            "python3 BagOfWords.py --source $source --username $username --medium $medium" *
            " --metric $metric --version $version"
        )
        run(Cmd(convert(Vector{String}, split(cmd))))
        outdir = joinpath(
            get_data_path("recommendations/$source/$username"),
            "alphas",
            medium,
            "BagOfWords",
            version,
            metric,
        )
        file = HDF5.h5open(joinpath(outdir, "predictions.h5"), "r")
        e = vec(read(file["predictions"]))
        if metric in ["watch", "plantowatch"]
            # make preds for regular items
            seen =
                get_raw_split(
                    "rec_training",
                    medium,
                    [:itemid],
                    nothing,
                    username,
                    source,
                ).itemid
            r = copy(e)
            r[seen.+1] .= 0
            r = r ./ sum(r)
            # make preds for plantowatch items
            ptw =
                get_split(
                    "rec_training",
                    "plantowatch",
                    medium,
                    [:itemid],
                    nothing,
                    username,
                    source,
                ).itemid
            p = copy(e)
            watched = setdiff(Set(seen), Set(ptw))
            p[watched.+1] .= 0
            p = p ./ sum(p)
            # combine preds
            e = copy(r)
            e[ptw.+1] .= p[ptw.+1]
        elseif metric in ["rating", "drop"]
            nothing
        else
            @assert false
        end
        close(file)
        model(userids, itemids) = [e[x+1] for x in itemids]
        write_alpha(
            model,
            medium,
            "$medium/BagOfWords/$version/$metric",
            REC_SPLITS,
            username,
            source,
        )
    end

    function runscript(username::String, source::String)
        version = "v1"
        outdir = joinpath(
            get_data_path("recommendations/$source/$username"),
            "alphas",
            "BagOfWords",
            version,
        )
        save_features(outdir, username, source)
        @sync for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
                Threads.@spawn compute_alpha(username, source, medium, metric, version)
            end
        end
        rm(outdir; recursive = true)
    end
end

runscript(ARGS...)