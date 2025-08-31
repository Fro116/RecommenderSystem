import JSON3
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")


function write_yaml(;prod::Bool)
    if prod
        nodes = 4
        finalcmds = ["./startup.sh"]
        name = "prod"
    else
        nodes = 1
        finalcmds = ["apt install tmux -y", "sleep 86400"]
        name = "dev"
    end
    image = "nvcr.io/nvidia/pytorch:25.08-py3"
    envvars = readlines("../../secrets/r2.auth.txt")
    script = [
        "export NUM_NODES=$nodes",
        "apt update",
        "apt install wget -y",
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh -O startup.sh",
        "chmod +x startup.sh",
        finalcmds...,
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    yaml = read("entrypoint.yaml", String)
    yaml = replace(yaml, "{IMAGE}" => image, "{ENTRYPOINT}" => entrypoint, "{NODES}" => nodes)
    fn = "../../data/training/$name.yaml"
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

function stop_sfcompute()
    scaleid = read("../../secrets/sfcompute.scale.txt", String)
    while true
        try
            run(`sf scale update $scaleid -n 0 -y`)
            return
        catch e
            logerror("failed to stop sfcompute $e")
            sleep(60)
        end
    end
end

function start_sfcompute(nodes::Int, gpuhour_price::Real)::Bool
    @assert nodes <= 4 && gpuhour_price <= 3
    scaleid = read("../../secrets/sfcompute.scale.txt", String)
    gpuhour_price = string(round(gpuhour_price, digits=2))
    ngpus = nodes * 8
    run(`sf scale update $scaleid -p $gpuhour_price -n $ngpus -d 30m -y`)
    cluster = get_active_sf_cluster()
    while isnothing(cluster)
        logerror("wating for cluster to startup")
        sleep(60)
        cluster = get_active_sf_cluster()
    end
    try
        cmds = [
            "sf clusters users add --cluster $cluster --user myuser",
            "sleep 60",
            "kubectl delete job --all --ignore-not-found",
            "cd ../../data/training",
            "kubectl apply -f causal.yaml",
            "kubectl apply -f masked.yaml",
        ]
        cmd = join(cmds, " && ")
        run(`sh -c $cmd`)
    catch e
        logerror("start_sfcompute: recieved error $e when connecting to cluster")
        stop_sfcompute()
        return false
    end
    true
end

function wait_sfcompute()
    list_tag = read("../../data/training/list_tag", String)
    finished = false
    while !finished
        finished = true
        print("rclone --retries=10 ls r2:rsys/database/training/$list_tag/transformer.$modeltype.finished")
        success = read(
            `rclone --retries=10 ls r2:rsys/database/training/$list_tag/transformer.$modeltype.finished`,
            String,
        )
        if isempty(success)
            finished = false
            break
        end
        if !finished
            logtag("[RUNGPU]", "waiting for models to finish")
            sleep(600)
        end        
    end
    stop_sfcompute()
end

function pretrain()
    num_nodes = 4
    gpuhour_price = 2.5
    write_yaml(prod=true)
    success = start_sfcompute(num_nodes, gpuhour_price)
    if !sfcompute_success
        logerror("sfcompute training failed")
        return
    end
    wait_sfcompute()
    stop_sfcompute()
    run(`rclone --retries=10 copyto ../../data/training/list_tag r2:rsys/database/training/latest`)
end

write_yaml(prod=false)
pretrain()