import Dates
import JSON3
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

function get_active_sf_cluster()
    read_json(cmd) = JSON3.read(replace(read(cmd, String), r"(\w+): " => s"\"\1\": "))
    for _ in 1:3
        try
            contracts = read_json(`sf contracts list --json`)
            cluster_id = first(contracts)["cluster_id"]
            clusters = read_json(`sf clusters list --json`)
            for x in clusters
                if x["contract"]["cluster_id"] == cluster_id
                    return x["name"]
                end
            end
        catch e
            logerror("get_active_sf_cluster $e")
        end
        sleep(60)
    end
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
    runtime_mins = 240
    runtime_mins -= 60 - Dates.minute(Dates.now())
    while runtime_mins > 0
        runtime_mins -= 60
        h += 1
    end
    @assert h <= 5
    try
        run(`sf buy -d $(h)hr -n 8 -p 3 -y`)
        logtag("RUN", "started sfcompute")
        cluster = get_active_sf_cluster()
        @assert !isnothing(cluster)
        run(`sf clusters users add --cluster $cluster --user myuser`)
        run(`kubectl apply -f $fn`)
        sleep(h * 3600)
        return true
    catch e
        logerror("sfcompute error $e")
        return false
    end
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
        runtime_mins = 240
        sleep(runtime_mins * 60)
        return true
    catch
        return false
    end
end

function start_gpu()
    while true
        success = start_sfcompute()
        if success
            return
        end
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
    latest = read("../../data/training/latest", String)
    success = read(
        `rclone --retries=10 ls r2:rsys/database/training/$latest/bagofwords.1.drop.pt`,
        String
    )
    if isempty(success)
        logerror("gpu training failed")
        return
    end
    cmd = "rclone --retries=10 copyto ../../data/training/latest r2:rsys/database/training/latest"
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
