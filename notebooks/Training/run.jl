import JSON3
include("../julia_utils/stdout.jl")

function get_gpu_args()
    image = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
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
        contracts = read_json(`sf contracts list --json`)
        cluster_id = first(contracts)["cluster_id"]
        clusters = read_json(`sf clusters list --json`)
        for x in clusters
            if x["contract"]["cluster_id"] == cluster_id
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
        h = 8
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
        run(`kubectl apply -f $fn`)
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
    run(`julia baseline.jl`)
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
