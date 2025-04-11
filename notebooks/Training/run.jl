import JSON3
include("../julia_utils/stdout.jl")

function get_gpu_args()
    image = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "apt update",
        "apt install wget -y",
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh -O startup.sh",
        "chmod +x startup.sh",
        "./startup.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    image, entrypoint
end

function get_active_sf_cluster()
    read_json(cmd) = JSON3.read(replace(read(cmd, String), r"(\w+): " => s"\"\1\": "))
    try
        clusters = read_json(`sf clusters list --json`)
        for x in clusters
            if x["contract"]["status"] == "active"
                return x["name"]
            end
        end
    catch e
        return nothing
    end
end

function start_sfcompute(gpuhour_price)
    image, entrypoint = get_gpu_args()
    yaml = read("entrypoint.yaml", String)
    yaml = replace(yaml, "{IMAGE}" => image, "{ENTRYPOINT}" => entrypoint)
    fn = "../../data/training/prod.yaml"
    open(fn, "w") do f
        write(f, yaml)
    end
    if !isnothing(get_active_sf_cluster())
        logerror("start_sfcompute already launched cluster")
        return false
    end
    should_retry = true
    try
        h = 7
        gpuhour_price = string(round(gpuhour_price, digits=2))
        run(`sf buy -d $(h)hr -n 8 -p $gpuhour_price -y`)
        logtag("RUN", "started sfcompute")
        sleep(60)
        cluster = get_active_sf_cluster()
        if isnothing(cluster)
            return should_retry # order did not get filled
        end
        should_retry = false
        run(`sf clusters users add --cluster $cluster --user myuser`)
        cmds = [
            "sleep 60",
            "kubectl delete jobs `kubectl get jobs -o custom-columns=:.metadata.name`",
            "cd ../../data/training",
            "kubectl apply -f prod.yaml",
        ]
        cmd = join(cmds, " && ")
        run(`sh -c $cmd`)
        sleep(h * 3600)
        return should_retry
    catch e
        logerror("start_sfcompute error $e")
        return should_retry
    end
end

function start_gpu()
    price = 3
    while true
        should_retry = start_sfcompute(price)
        if !should_retry
            return
        end
        logtag("RUN", "waiting for available gpus...")
        sleep(600)
    end
end

function pretrain()
    run(`julia import_data.jl`)
    for m in [0, 1]
        run(`julia media_relations.jl $m`)
    end
    for metric in ["rating"]
        run(`julia baseline.jl $metric`)
    end
    run(`julia -t auto bagofwords.jl --pretrain`)
    run(`julia -t auto transformer.jl`)
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

pretrain()
