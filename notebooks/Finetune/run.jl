include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function get_gpu_args()
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Finetune/entrypoint.sh",
        "chmod +x entrypoint.sh",
        "./entrypoint.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    image, entrypoint
end

function start_runpod(gpuname)
    image, entrypoint = get_gpu_args()
    podname = "prod_finetune"
    create = [
        "runpodctl create pod",
        "--name $podname",
        "--gpuType '$gpuname'",
        "--gpuCount 1",
        "--imageName '$image'",
        "--volumePath /data",
        "--volumeSize 20",
        "--secureCloud",
        "--args 'bash -c \"$entrypoint\"'",
    ]
    create = join(create, " ")
    wait = "sleep 60 && while runpodctl get pod | grep -w $podname > /dev/null 2>&1; do sleep 5; done"
    cmds = [
        create,
        wait
    ]
    cmd = join(cmds, " && ")
    try
        run(`sh -c $cmd`)
        return true
    catch
        return false
    end
end

function start_gpu()
    while true
        gputypes = ["NVIDIA GeForce RTX 4090"]
        for gpuname in gputypes
            success = start_runpod(gpuname)
            if success
                return
            end
        end
        logtag("RUN", "waiting for available gpus...")
        sleep(600)
    end
end

function train()
    run(`julia import_data.jl`)
    run(`julia ../Training/bagofwords.jl --finetune`)
    start_gpu()
    # for p in ["Fetch", "Read"]
    #     cmd = "cd $p && julia package.jl"
    #     run(`sh -c $cmd`)
    # end
end

@scheduled "FINETUNE" "15:00" @handle_errors train()
