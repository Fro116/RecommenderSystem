import Dates
import HTTP
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")
first_run::Bool = true

function update_local_server(task)
    username, password, project = split(read("../secrets/docker.auth.txt", String), "\n")
    name = "server"
    tag = "latest"
    cmds = ["(echo '$password' | docker login -u $username --password-stdin)"]
    if task == "pull"
        push!(cmds, "docker pull $username/$project-$name:$tag")
        cmd = join(cmds, " && ")
        run(`sh -c $cmd`)
    elseif task == "run"
        push!(cmds, "docker run --rm --runtime=nvidia --gpus=all --network=host --name $name $username/$project-$name:$tag")
        cmd = join(cmds, " && ")
        Threads.@spawn @handle_errors run(`sh -c $cmd`)
        ready = false
        while !ready
            try
                HTTP.get("http://localhost:8080/ready")
                ready = true
            catch
                logtag("INFERENCE", "waiting for server to startup")
                sleep(60)
            end
        end
    else
        @assert false
    end
end

function stop_local_server()
    name = "server"
    try
        run(`docker stop $name`)
    catch e
        if !first_run
            logerror("could not stop $name: $e")
        end
    end
end

function resize_remote_server(size)
    logtag("INFERENCE", "resizing remote servers to $size instances")
    project = read("../secrets/gcp.project.txt", String)
    region = read("../secrets/gcp.region.txt", String)
    group = read("../secrets/gcp.igname.txt", String)
    t = Int(round(time()))
    cmds = [
        "gcloud auth login --cred-file=../secrets/gcp.auth.json --quiet",
        "gcloud compute instance-groups managed resize $group --project=$project --region=$region --size=$size"
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    if size != 0
        sleep(25 * 60)
    end
end

function update()
    global first_run
    update_local_server("pull")
    if !first_run
        resize_remote_server(1)
    end
    ts = time()
    stop_local_server()
    update_local_server("run")
    logtag("INFERENCE", "server started in $(time() - ts) seconds")
    resize_remote_server(0)
    run(`docker system prune -af --filter until=24h`)
    first_run = false
end

update()
@scheduled "INFERENCE" "05:00" @handle_errors update()
