#   Neural
#   ≡≡≡≡≡≡≡≡
# 
#     •  See the corresponding file in ../../TrainingAlphas for more
#        details

import NBInclude: @nbinclude
import Memoize: @memoize
if !@isdefined NEURAL_IFNDEF
    NEURAL_IFNDEF=true
    source_name = "Neural"
    
    using CUDA
    using Flux
    import Functors: @functor
    import NBInclude: @nbinclude
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Hyperparameters.ipynb")
    @nbinclude("../../TrainingAlphas/Neural/Helpers/Models.ipynb")
    @nbinclude("Data.ipynb")
    Logging.disable_logging(Logging.Warn)
    
    @memoize read_params_memoized(source) = read_params(source)
    jobs = []
    for task in ALL_TASKS
        for content in ["explicit", "implicit"]
            push!(jobs, ("$task/Neural$(uppercasefirst(content))AutoencoderUntuned", task))
        end
        push!(jobs, ("$task/NeuralExplicitItemCFUntuned", task))
        push!(jobs, ("$task/NeuralImplicitEaseUntuned", task))
    end
    for j in jobs
        read_params_memoized(j[1])
    end
        
    function compute_alpha(source, task)
        params = read_params_memoized(source)
        m = params["m"]
        inputs = get_recommendee_inputs(params["hyp"], task)
        activation = params["hyp"].implicit ? softmax : identity
        preds = vec(activation(m(inputs)))
        write_recommendee_alpha(preds, source)
    end;
end

# TODO parallelize
username = ARGS[1]
for j in jobs
    compute_alpha(j...)
end