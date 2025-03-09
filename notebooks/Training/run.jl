import Dates
import JSON3
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")

function get_gpu_args()
    image = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh",
        "chmod +x entrypoint.sh",
        "./entrypoint.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    image, entrypoint
end

function has_active_sf_cluster()
    read_json(cmd) = JSON3.read(replace(read(cmd, String), r"(\w+): " => s"\"\1\": "))
    contracts = read_json(`sf contracts list --json`)
    return !isempty(contracts)
end

function get_active_sf_cluster()
    read_json(cmd) = JSON3.read(replace(read(cmd, String), r"(\w+): " => s"\"\1\": "))
    contracts = read_json(`sf contracts list --json`)
    cluster_id = first(contracts)["cluster_id"]
    clusters = read_json(`sf clusters list --json`)
    for x in clusters
        if x["contract"]["cluster_id"] == cluster_id
            return x["name"]
        end
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
    started_gpu = false
    h = 8
    if has_active_sf_cluster()
        logerror("start_sfcompute already launched")
        return false
    end
    try
        run(`sf buy -d $(h)hr -n 8 -p 3 -y`)
        started_gpu = true
        logtag("RUN", "started sfcompute")
        sleep(60)
        cluster = get_active_sf_cluster()
        @assert !isnothing(cluster)
        run(`sf clusters users add --cluster $cluster --user myuser`)
        run(`kubectl apply -f $fn`)
        sleep(h * 3600)
        return started_gpu
    catch e
        logerror("start_sfcompute error $e")
        return started_gpu
    end
end

function start_gpu()
    while true
        success = start_sfcompute()
        if success
            return
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
        `rclone --retries=10 ls r2:rsys/database/training/$latest/bagofwords.1.rating.pt`,
        String,
    )
    if isempty(success)
        logerror("gpu training failed")
        return
    end
    cmd = "rclone --retries=10 copyto ../../data/training/latest r2:rsys/database/training/latest"
    run(`sh -c $cmd`)
    cleanup =
        raw"rclone lsd r2:rsys/database/training/ | sort | head -n -2 | awk '{print $NF}' | xargs -I {} rclone purge r2:rsys/database/training/{}"
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
