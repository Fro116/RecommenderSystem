import JSON3
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")

function write_entrypoint_yaml(modeltype, nodes)
    image = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "export NUM_NODES=$nodes",
        "export MODELTYPE=$modeltype",
        "apt update",
        "apt install wget -y",
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh -O startup.sh",
        "chmod +x startup.sh",
        "./startup.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    yaml = read("entrypoint.yaml", String)
    yaml = replace(yaml, "{IMAGE}" => image, "{ENTRYPOINT}" => entrypoint, "{NODES}" => nodes, "{MODELTYPE}" => modeltype)
    fn = "../../data/training/$modeltype.yaml"
    open(fn, "w") do f
        write(f, yaml)
    end
end

function read_json(cmd::Cmd)
    text = try
        read(cmd, String)
    catch
        logerror("read_json: could not run $cmd")
        return nothing
    end
    try
        return JSON3.read(replace(text, r"(\w+): " => s"\"\1\": "))
    catch
        logerror("read_json: could not parse $text")
        return nothing
    end
end

function get_active_sf_cluster()
    clusters = read_json(`sf clusters list --json`)
    if isnothing(clusters)
        return nothing
    end
    for x in clusters
        if x["contract"]["status"] == "active"
            return x["name"]
        end
    end
end

# returns true on success
function start_sfcompute(gpuhour_price::Real)::Bool
    nodes = 4
    runtime_hours = 12
    for modeltype in ["causal", "masked"]
        write_entrypoint_yaml(modeltype, nodes)
    end
    if !isnothing(get_active_sf_cluster())
        logerror("start_sfcompute already launched cluster")
        return false
    end
    gpuhour_price = string(round(gpuhour_price, digits=2))
    try
        logtag("RUN", "ordering sfcompute for $runtime_hours hours at price \$$gpuhour_price/gpuhour")
        run(`sf buy -d $(runtime_hours)hr -n $(8*nodes) -p $gpuhour_price -y`)
        logtag("RUN", "starting sfcompute...")
    catch e
        logerror("start_sfcompute: error $e")
        return false
    end
    sleep(60) # wait for cluster to spin up
    cluster = get_active_sf_cluster()
    if isnothing(cluster)
        logerror("start_sfcompute: no cluster found. Perhaps the order was not filled.")
        return false
    end
    try
        cmds = [
            "sf clusters users add --cluster $cluster --user myuser",
            "sleep 60",
            "kubectl delete jobs `kubectl get jobs -o custom-columns=:.metadata.name`",
            "cd ../../data/training",
            "kubectl apply -f causal.yaml",
            "kubectl apply -f masked.yaml",
        ]
        cmd = join(cmds, " && ")
        run(`sh -c $cmd`)
        sleep(runtime_hours * 3600)
    catch e
        logerror("start_sfcompute: recieved error $e when connecting to cluster")
        return false
    end
    true
end

function pretrain(datetag::AbstractString)
    run(`julia import_data.jl $datetag`)
    for m in [0, 1]
        run(`julia media_relations.jl $m`)
    end
    run(`julia -t auto transformer.jl`)
    # start_sfcompute(3.00)
    # if !sfcompute_success
    #     logerror("sfcompute training failed")
    #     return
    # end
    # list_tag = read("../../data/training/list_tag", String)
    # for f in ["transformer.causal.pt", "transformer.masked.pt"]
    #     success = read(
    #         `rclone --retries=10 ls r2:rsys/database/training/$list_tag/$f`,
    #         String,
    #     )
    #     if isempty(success)
    #         logerror("gpu training failed")
    #         return
    #     end
    # end
    # cmd = "rclone --retries=10 copyto ../../data/training/list_tag r2:rsys/database/training/latest"
    # run(`sh -c $cmd`)
    # cleanup =
    #     raw"rclone lsd r2:rsys/database/training/ | sort | head -n -2 | awk '{print $NF}' | xargs -I {} rclone purge r2:rsys/database/training/{}"
    # run(`sh -c $cleanup`)
end

pretrain(ARGS[1])
