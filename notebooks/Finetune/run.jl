function torchrun(cmd)
    cmd = "cd ../Training && $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)"
    run(`sh -c $cmd`)
end

function finetune(list_tag::AbstractString)
    run(`julia import_data.jl $list_tag`)
    run(`julia transformer.jl`)
    for modeltype in ["masked", "causal"]
        for m in [0, 1]
            torchrun("torchrun --standalone --nproc_per_node=1 transformer.py --datadir ../../data/finetune --finetune ../../data/finetune/transformer.$modeltype.pt --modeltype $modeltype --finetune_medium $m")
        end
    end
    run(`python register.py`)
    run(`julia regress.jl`)
    run(`julia pairwise.jl`)
    for app in ["Embed", "Compute"]
        cmd = "cd ../Package/$app && julia package.jl"
        run(`sh -c $cmd`)
    end
end

finetune(ARGS[1])
