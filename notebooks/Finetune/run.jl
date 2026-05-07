function torchrun(cmd)
    cmd = "cd ../Training && $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)"
    run(`sh -c $cmd`)
end

function finetune(list_tag::AbstractString)
    run(`julia import_data.jl $list_tag`)
    run(`julia transformer.jl`)
    for m in [0, 1]
        for metric in ["rating", "watch"]
            torchrun("torchrun --standalone --nproc_per_node=1 transformer.py --datadir ../../data/finetune --finetune ../../data/finetune/transformer.masked.pt --finetune_medium $m --finetune_metric $metric")
        end
    end
    run(`python register.py`)
    run(`julia regress.jl`)
    run(`julia pairwise.jl`)
    cmd = "cd ../Package/Server && julia package.jl"
    run(`sh -c $cmd`)
end

finetune(ARGS[1])
