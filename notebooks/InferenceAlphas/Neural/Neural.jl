import NBInclude: @nbinclude
import Memoize: @memoize
if !@isdefined NEURAL_IFNDEF
    NEURAL_IFNDEF=true
    source_name = "Neural"
    @nbinclude("../Alpha.ipynb")    
    using CUDA
    using Flux
    import Functors: @functor
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Hyperparameters.ipynb")
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Models.ipynb")
    
    function explicit_inputs(input_alphas::Vector{String}, task::String, medium::String)
        df = get_recommendee_split("explicit", medium)
        residual = read_recommendee_alpha(input_alphas, task, "explicit", medium, false)
        inputs = zeros(Float32, num_items(medium))
        inputs[df.item] .= df.rating - residual.rating
        inputs
    end
    
    function implicit_inputs(medium::String)
        df = get_recommendee_split("implicit", medium)
        inputs = zeros(Float32, num_items(medium))
        inputs[df.item] .= df.rating
        inputs
    end
    
    function universal_inputs(input_alphas::Vector{String}, task::String)
        @assert length(input_alphas) == length(ALL_MEDIUMS)
        inputs = []
        for i in 1:length(ALL_MEDIUMS)
            push!(inputs, explicit_inputs(input_alphas[i:i], task, ALL_MEDIUMS[i]))
        end
        for x in ALL_MEDIUMS
            push!(inputs, implicit_inputs(x))
        end
        reduce(vcat, inputs)
    end
    
    function get_recommendee_inputs(hyp, task)
        if hyp.input_data == "universal"
            return universal_inputs(hyp.input_alphas, task)
        else
            @assert false
        end
    end
    
    @memoize read_params_memoized(source) = read_params(source)
    function compute_alpha(name, task, medium)
        name = "$medium/$task/$name"
        params = read_params_memoized(name)
        m = params["m"]
        inputs = get_recommendee_inputs(params["hyp"], task)
        activation = params["hyp"].implicit ? softmax : identity
        preds = vec(activation(m(inputs)))
        write_recommendee_alpha(preds, medium, name)
    end;
end

username = ARGS[1]
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_alpha("NeuralExplicitUniversalUntuned", task, medium)
        compute_alpha("NeuralImplicitUniversalUntuned", task, medium)
    end
end