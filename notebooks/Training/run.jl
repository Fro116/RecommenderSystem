import Dates
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function get_gpu_args()
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh",
        "chmod +x entrypoint.sh",
        "./entrypoint.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    image, entrypoint
end

function start_sfcompute()
    image, entrypoint = get_gpu_args()
    yaml = read("entrypoint.yaml", String)
    yaml = replace(yaml, "{IMAGE}" => image, "{ENTRYPOINT}" => entrypoint)
    fn = "../../data/training/prod.yaml"
    open(fn, "w") do f
        write(f, yaml)
    end
    h = 0
    runtime_mins = 150
    runtime_mins -= 60 - Dates.minute(Dates.now())
    while runtime_mins > 0
        runtime_mins -= 60
        h += 1
    end
    @assert h <= 5
    # TODO get cluster programatically
    cmds = [
        "sf buy -d $(h)hr -n 8 -p 2.50",
        "sf clusters users add --cluster alamo --user myuser",
        "kubectl apply -f prod.yaml"
    ]
    cmd = join(cmds, " && ")
    run("sh -c $cmd")
end

function start_runpod(gpuname)
    image, entrypoint = get_gpu_args()
    podname = "prod_train"
    create = [
        "runpodctl create pod",
        "--name $podname",
        "--gpuType '$gpuname'",
        "--gpuCount 8",
        "--imageName '$image'",
        "--containerDiskSize 128",
        "--secureCloud",
        "--args 'bash -c \"$entrypoint\"'",
    ]
    create = join(create, " ")
    cmds = [
        create,
        "sleep 300",
        "runpodctl get pod | grep -w $podname | grep -w RUNNING",
    ]
    cmd = join(cmds, " && ")
    try
        run(`sh -c $cmd`)
        logtag("RUN", "started runpod with command $cmd")
        cmd = "runpodctl get pod | grep -w $podname | grep -w RUNNING"
        while true
            try
                run(`sh -c $cmd`)
                logtag("RUN", "waiting for runpod to finish")
                sleep(3600)
            catch
                break
            end
        end
        return true
    catch
        return false
    end
end

function start_gpu()
    while true
        # TODO reevaluate sfcompute
        gputypes = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200"]
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

function pretrain()
    run(`julia import_data.jl`)
    run(`julia baseline.jl`)
    run(`julia -t auto bagofwords.jl --pretrain`)
    start_gpu()
    cmd = replace(
        "rclone --retries=10 copyto {INPUT} r2:rsys/database/training/{OUTPUT}",
        "{INPUT}" => "../../data/training/latest",
        "{OUTPUT}" =>  "latest",
    )
    run(`sh -c $cmd`)
    cleanup = raw"rclone lsd r2:rsys/database/training/ | sort | head -n -2 | awk '{print $NF}' | xargs -I {} rclone purge r2:rsys/database/training/{}"
    run(`sh -c $cleanup`)
end

function train()
    if Dates.dayofmonth(Dates.now()) in [1, 9, 15, 22]
        pretrain()
    end
    cmd = "cd ../Finetune && julia run.jl"
    run(`sh -c $cmd`)
end

@scheduled "TRAIN" "7:00" @handle_errors train()
