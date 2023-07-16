import NBInclude: @nbinclude
import Memoize: @memoize
if !@isdefined NEURAL_IFNDEF
    NEURAL_IFNDEF = true
    source_name = "Neural"
    @nbinclude("../Alpha.ipynb")
    using CUDA
    using Flux
    import Functors: @functor
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Hyperparameters.ipynb")
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Models.ipynb")

    function dropitem(df::RatingsDataset, exclude)
        if df.medium == exclude[:medium]
            return filter(df, df.item .!= exclude[:item])
        else
            return df
        end
    end

    function explicit_inputs(
        input_alphas::Vector{String},
        task::String,
        medium::String,
        exclude,
    )
        df = dropitem(get_recommendee_split("explicit", medium), exclude)
        residual = dropitem(
            read_recommendee_alpha(input_alphas, task, "explicit", medium, false),
            exclude,
        )
        inputs = zeros(Float32, num_items(medium))
        inputs[df.item] .= df.rating - residual.rating
        inputs
    end

    function implicit_inputs(medium::String, exclude)
        df = dropitem(get_recommendee_split("implicit", medium), exclude)
        inputs = zeros(Float32, num_items(medium))
        inputs[df.item] .= df.rating
        inputs
    end

    function universal_inputs(input_alphas::Vector{String}, task::String, exclude)
        @assert length(input_alphas) == length(ALL_MEDIUMS)
        inputs = []
        for i = 1:length(ALL_MEDIUMS)
            push!(inputs, explicit_inputs(input_alphas[i:i], task, ALL_MEDIUMS[i], exclude))
        end
        for x in ALL_MEDIUMS
            push!(inputs, implicit_inputs(x, exclude))
        end
        reduce(vcat, inputs)
    end

    function get_recommendee_inputs(hyp, task, exclude)
        if hyp.input_data == "universal"
            return universal_inputs(hyp.input_alphas, task, exclude)
        else
            @assert false
        end
    end

    @memoize read_params_memoized(source) = read_params(source)
    function compute_alpha(name, task, medium)
        name = "$medium/$task/$name"
        params = read_params_memoized(name)
        excludes = []
        for m in ALL_MEDIUMS
            items = vcat([get_recommendee_split(x, m).item for x in ["implicit", "ptw"]]...)
            for i in items
                push!(excludes, (item = i, medium = m))
            end
        end
        pushfirst!(excludes, (item=0, medium=medium))
        inputs = []
        for exclude in excludes
            push!(inputs, get_recommendee_inputs(params["hyp"], task, exclude))
        end
        m = params["m"]
        x = hcat(inputs...)
        activation = params["hyp"].implicit ? softmax : identity
        preds = activation(m(x))
        write_recommendee_alpha(preds[:, 1], medium, name)
        write_recommendee_params(Dict("excludes" => excludes, "alpha" => preds), name)
    end
end

username = ARGS[1]
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_alpha("NeuralExplicitUniversalUntuned", task, medium)
        compute_alpha("NeuralImplicitUniversalUntuned", task, medium)
    end
end