include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")



function pretrain()
    run(`julia import_data.jl`)
    run(`julia baseline.jl`)
    run(`julia bagofwords.jl`)
    for medium in [0 1]
        for metric in ["rating", "watch", "drop", "plantowatch"]
            run(
                `python ../Training/bagofwords.py
                     --datadir ../../data/training
                     --medium $medium
                     --metric $metric
                     --finetune ../../data/training/bagofwords.$medium.$metric.jl`
           )
        end
    end
    for p in ["Fetch", "Read"]
        cmd = "cd $p && julia package.jl"
        run(`sh -c $cmd`)
    end
end

@scheduled "FINETUNE" "15:00" @handle_errors train()
