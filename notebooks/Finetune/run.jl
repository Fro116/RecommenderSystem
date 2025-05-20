function torchrun(cmd)
    cmd = "cd ../Training && $cmd || (sleep 10 && $cmd) || (sleep 60 && $cmd)"
    run(`sh -c $cmd`)
end

function blue_green_deploy()
    active = read("../../data/finetune/bluegreen", String)
    disable = Dict("blue" => "green", "green" => "blue")[active]
    region = read("../../secrets/gcp.region.txt", String)
    cmds = [
        "gcloud auth login --cred-file=../../secrets/gcp.auth.json --quiet",
        "gcloud beta run services update embed-$disable --scaling=0 --region $region",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function finetune(list_tag::AbstractString)
    run(`julia import_data.jl $list_tag`)
    run(`julia transformer.jl`)
    for modelype in ["retrieval", "ranking"]
        for m in [0, 1]
            torchrun("torchrun --standalone --nproc_per_node=1 transformer.py --datadir ../../data/finetune --finetune ../../data/finetune/transformer.pt --modeltype $modeltype --finetune_medium $m")
        end
    end
    # run(`python register.py`)
    # run(`julia regress.jl`)
    # for app in ["Embed", "Compute"]
    #     cmd = "cd ../Package/$app && julia package.jl"
    #     run(`sh -c $cmd`)
    # end
    # blue_green_deploy()
end

finetune(ARGS[1])
