import NBInclude: @nbinclude
import Setfield: @set

if !@isdefined BAGOFWORDS_IFNDEF
    BAGOFWORDS_IFNDEF = true

    source_name = "BagOfWords"
    import HDF5
    import JSON
    import SparseArrays: AbstractSparseArray, sparse
    @nbinclude("../Alpha.ipynb")

    function explicit_inputs(task::String, medium::String, residual_alphas::Vector{String})
        df = get_recommendee_split("explicit", medium)
        df = RatingsDataset(
            user = df.user,
            item = df.item,
            rating = df.rating .-
                     read_recommendee_alpha(
                residual_alphas,
                task,
                "explicit",
                medium,
                false,
            ).rating,
            medium = medium,
        )
        sparse(df)
    end

    function implicit_inputs(medium::String)
        df = get_recommendee_split("implicit", medium)
        sparse(df)
    end

    function get_epoch_inputs(task::String, residual_alphas::Vector{String})
        @assert length(residual_alphas) == length(ALL_MEDIUMS)
        inputs = []
        for i = 1:length(ALL_MEDIUMS)
            push!(inputs, explicit_inputs(task, ALL_MEDIUMS[i], residual_alphas[i:i]))
        end
        for x in ALL_MEDIUMS
            push!(inputs, implicit_inputs(x))
        end
        reduce(vcat, inputs)
    end

    function save_features(task, outdir)
        mkpath(outdir)
        filename = joinpath(outdir, "inference.h5")
        d = Dict{String,Any}()
        X = get_epoch_inputs(task, ["$x/$task/ExplicitUserItemBiases" for x in ALL_MEDIUMS])
        record_sparse_array!(d, "inputs", X)
        d["users"] = [1]
        HDF5.h5open(filename, "w") do file
            for (k, v) in d
                write(file, k, v)
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

    function compute_alpha(username, task, medium, content, version)
        outdir = joinpath(
            recommendee_alpha_basepath(),
            medium,
            task,
            "BagOfWords",
            content,
            version,
        )
        save_features(task, outdir)
        cmd = (
            "python3 BagOfWords.py --username $username --medium $medium" *
            " --task $task --content $content --version $version"
        )
        run(Cmd(convert(Vector{String}, split(cmd))))
        file = HDF5.h5open(joinpath(outdir, "predictions.h5"), "r")
        preds = read(file["predictions"])
        savedir = joinpath(medium, task, "BagOfWords", content, version)
        write_recommendee_alpha(vec(preds), medium, savedir)
    end
end

username = ARGS[1]
version = "v1"
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        for content in ["implicit", "explicit"]
            compute_alpha(username, task, medium, content, version)
        end
    end
end