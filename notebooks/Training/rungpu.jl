import JSON3
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
const datadir = "../../data/training"

function write_entrypoint(num_gpus)
    envvars = read("../../secrets/r2.auth.txt", String)
    script = read("entrypoint.sh", String)
    script = replace(script, "{{ENVVARS}}" => envvars, "{{NUM_GPUS}}" => num_gpus)
    fn = "$datadir/entrypoint.vastai.sh"
    open(fn, "w") do f
        write(f, script)
    end
end

function get_balance()
    try
        json = read(`vastai show user --raw`, String)
        return JSON3.parse(json)[:credit]
    catch e
        logerror("$e")
        return 0
    end
end

function provision_instance()
    get_instances() = copy(JSON3.parse(read(`vastai show instances --raw`, String)))
    @assert isempty(get_instances()) "instance already exists"
    for gpu_config in ["8xB200", "4xB200", "8xH100"]
        node_price = Dict(
            "8xB200" => 32,
            "4xB200" => 16,
            "8xH100" => 24,
        )[gpu_config]
        duration = Dict(
            "8xB200" => 25,
            "4xB200" => 50,
            "8xH100" => 50,
        )[gpu_config]
        num_gpus, gpu_type = split(gpu_config, "x")
        gpu_query = Dict(
            "B200" => """gpu_name in ["B200"]""",
            "H100" => """gpu_name in ["H200", "H100_SXM"]""",
        )[gpu_type]
        balance = get_balance()
        offers = copy(
            JSON3.parse(
                read(
                    `vastai search offers num_gpus=$num_gpus $gpu_query 'cuda_max_good>=12.8' datacenter=true verified=true --raw`,
                    String,
                ),
            ),
        )
        sort!(offers, by = x -> x[:dph_total])
        success = false
        for x in offers
            offer_id = string(x[:id])
            if x[:dph_total] > node_price
                logerror(
                    "skipping $instance_id because price $(x[:dph_total]) > $node_price",
                )
                continue
            end
            if balance < x[:dph_total] * duration
                logerror("insufficient funds: $balance < $(x[:dph_total]) * $duration")
                continue
            end
            disk_size = 96
            logtag("RUNGPU", "provisioning $gpu_config instance for price $(x[:dph_total])")
            create_cmd = """vastai create instance $offer_id --image vastai/pytorch:cuda-12.8.1-auto --env '-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard"' --onstart-cmd 'entrypoint.sh' --disk $disk_size --jupyter --ssh --direct"""
            run(`sh -c $create_cmd`)
            sleep(60)
            write_entrypoint(num_gpus)
            instance = only(get_instances())
            return string(instance[:id]), duration
        end
    end
    nothing
end

function launch_job(instance_id::String, duration::Real)
    function get_instance_status(instance_id)
        instance = copy(JSON3.parse(read(`vastai show instance $instance_id --raw`, String)))
        instance[:actual_status]
    end
    run(`vastai start instance $instance_id`)
    logtag("RUNGPU", "waiting for instance to reach 'running' state ...")
    started = false
    max_wait = time() + 300
    while time() < max_wait
        if get_instance_status(instance_id) == "running"
            started = true
            break
        end
        sleep(30)
    end
    if !started
        logerror("vastai failed to start $instance_id")
        run(`vastai stop instance $instance_id`)
        return false
    end
    ssh_url = strip(read(`vastai ssh-url $instance_id`, String))
    s = replace(ssh_url, r"^ssh://" => "")
    host, port = split(s, ":", limit = 2)
    scp_command = `scp -P $port -o StrictHostKeyChecking=accept-new $datadir/entrypoint.vastai.sh $host:/workspace/startup.sh`
    ssh_command = `ssh -o StrictHostKeyChecking=accept-new -p $port $host 'tmux new -s startup -d "cd /workspace && chmod +x startup.sh && ./startup.sh &> log"'`
    logtag("RUNGPU", "running $scp_command")
    run(scp_command)
    logtag("RUNGPU", "running $ssh_command")
    run(ssh_command)
    logtag("RUNGPU", "waiting for job completion")
    finished = false
    max_wait = time() + duration * 3600
    while time() < max_wait
        if get_instance_status(instance_id) != "running"
            finished = true
            break
        end
        sleep(600)
    end
    run(`vastai stop instance $instance_id`)
    true
end

function destroy_instance(instance_id)
    run(`vastai destroy instance $instance_id`)
end

function launch_job()
    instance = nothing
    max_wait = time() + 3600 * 8
    while time() < max_wait
        instance = provision_instance()
        if !isnothing(instance)
            break
        end
        timeout = 600
        logerror("failed to provision gpu instance, retrying in $timeout seconds")
        sleep(timeout)
    end
    if isnothing(instance)
        logerror("failed to provision gpu instance, exiting")
        return
    end
    instance_id, duration = instance
    try
        launch_job(instance_id, duration)
    finally
        destroy_instance(instance_id)
    end
end

launch_job()