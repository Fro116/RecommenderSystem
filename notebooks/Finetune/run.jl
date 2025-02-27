function finetune()
    run(`julia import_data.jl`)
    run(`julia ../Training/bagofwords.jl --finetune`)
    for m in [0, 1]
        for metric in ["rating", "watch", "plantowatch", "drop"]
            cmd = "torchrun --standalone --nproc_per_node=1 bagofwords.py --datadir ../../data/finetune --medium $m --metric $metric --finetune ../../data/finetune/bagofwords.$m.$metric.pt"
            cmd = "cd ../Training && $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)"
            run(`sh -c $cmd`)
        end
    end
    run(`python register.py`)
    run(`julia regress.jl`)
    cmd = "cd ../Package/Embed && julia package.jl"
    run(`sh -c $cmd`)
end

finetune()
