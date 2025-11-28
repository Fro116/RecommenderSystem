import JSON3
include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
const datadir = "../../data/training"

function write_entrypoint()
    envvars = read("../../secrets/r2.auth.txt", String)
    script = read("entrypoint.sh", String)
    script = replace(script, "{{ENVVARS}}" => envvars)
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

function run_vast_instance(instance_id::String, duration::Real)
    run(`vastai start instance $instance_id`)
    println("Waiting for instance to reach 'running' state ...")
    started = false
    for _ = 1:10
        output = read(`vastai show instance $instance_id`, String)
        if occursin("running", lowercase(output))
            println("Instance is running.")
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
    sleep(duration * 3600) # TODO intelligent sleep where we early stop if instance is dead
    run(`vastai stop instance $instance_id`)
    true
end

function launch_job()
    write_entrypoint()
    duration = 25
    node_price = 30
    balance = get_balance()
    instances = copy(JSON3.parse(read(`vastai show instances --raw`, String)))
    sort!(instances, by = x -> x[:dph_total])
    for x in instances
        instance_id = string(x[:id])
        if x[:dph_total] > node_price
            logerror("skipping $instance_id because price $(x[:dph_total]) > $node_price")
            continue
        end
        if balance < x[:dph_total] * duration
            logerror("insufficient funds: $balance < $(x[:dph_total]) * $duration")
            continue
        end
        success = run_vast_instance(instance_id, duration)
        if success
            return true
        end
    end
    false
end

launch_job()
