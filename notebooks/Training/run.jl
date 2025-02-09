import Dates
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function pretrain()
    run(`julia import_data.jl`)
    run(`julia baseline.jl`)
    run(`julia -t auto bagofwords.jl`)
    cmd = "cd ../../environment/gpu && chmod +x provision.sh && ./provision.sh"
    run(`sh -c $cmd`)
end

function finetune()
end

function train()
    if Dates.dayofmonth(Dates.now()) in [1, 15]
        pretrain()
    end
    finetune()
end

@scheduled "TRAIN" "7:00" @handle_errors train()
