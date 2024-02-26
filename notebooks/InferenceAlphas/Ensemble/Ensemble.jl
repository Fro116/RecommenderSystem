import NBInclude: @nbinclude
if !@isdefined ENSEMBLE_IFNDEF
    ENSEMBLE_IFNDEF = true
    import H5Zblosc
    import HDF5
    @nbinclude("../../TrainingAlphas/Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Ensemble/EnsembleInputs.ipynb")

    function compute_linear_alpha(metric::String, medium::String)
        alphas = get_ensemple_alphas(metric, medium)
        X = [get_raw_split("rec_inference", medium, [:userid], a).alpha for a in alphas]
        if metric in ["watch", "plantowatch"]
            push!(X, fill(1.0f0 / num_items(medium), length(X[1])))
        elseif metric == "drop"
            push!(X, fill(1.0f0, length(X[1])), fill(0.0f0, length(X[1])))
        end
        X = reduce(hcat, X)
        params = read_params("$medium/Linear/$metric", true)
        e = X * params["β"]
        model(userids, itemids) = [e[x+1] for x in itemids]
        write_alpha(model, medium, "$medium/Linear/$metric", REC_SPLITS)
    end

    function get_mle_features(medium::String)
        alphas = ["$medium/Linear/$metric" for metric in ALL_METRICS]
        N = length(get_raw_split("rec_inference", medium, [:userid], nothing).userid)
        T = Float32
        A = Matrix{T}(undef, N, length(alphas))
        for i = 1:length(alphas)
            x = get_raw_split("rec_inference", medium, [:userid], alphas[i]).alpha
            # normalize and make monotonic
            if alphas[i] == "$medium/Linear/rating"
                x = clamp.(x / 10, 0, 1)
            elseif alphas[i] in ["$medium/Linear/watch", "$medium/Linear/plantowatch"]
                nothing
            elseif alphas[i] == "$medium/Linear/drop"
                x = 1 .- x
            else
                @assert false
            end
            @assert minimum(x) >= 0 && maximum(x) <= 1
            A[:, i] = convert.(T, x)
        end
        collect(A')
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

    function compute_mle_alpha(name::String, medium::String)
        outdir = joinpath(
            get_data_path("recommendations/$(get_rec_usertag())"),
            "alphas",
            medium,
            "Ranking",
        )
        mkpath(outdir)
        filename = joinpath(outdir, "inference.h5")
        F = get_mle_features(medium)
        HDF5.h5open(filename, "w") do file
            file["features", blosc = 1] = F
        end
        cmd = `python3 Ranking.py --outdir $outdir --medium $medium`
        run(cmd)
        file = HDF5.h5open(joinpath(outdir, "predictions.h5"), "r")
        e = vec(read(file["predictions"]))
        close(file)
        model(userids, itemids) = [e[x+1] for x in itemids]
        write_alpha(model, medium, name, REC_SPLITS)
    end
end

username = ARGS[1]
source = ARGS[2]
for medium in ALL_MEDIUMS
    for metric in ALL_METRICS
        compute_linear_alpha(metric, medium)
    end
    compute_mle_alpha("$medium/Ranking", medium)
end