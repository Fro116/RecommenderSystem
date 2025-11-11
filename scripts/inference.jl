import Dates
import HTTP
include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/stdout.jl")
cd("../notebooks")
first_run::Bool = true

function docker_login()
    username, password, project = split(read("../secrets/docker.auth.txt", String), "\n")
    name = "server"
    tag = "latest"
    run(
        pipeline(
            `docker login --username $username --password-stdin`,
            stdin = IOBuffer(password),
            stdout = devnull,
            stderr = devnull,
        ),
    )
    "$username/$project-$name:$tag"
end

function is_image_up_to_date()
    function get_digest(output)
        try
            match(r"sha256:[0-9a-f]{64}", output).match
        catch e
            logerror(e)
            ""
        end
    end
    name = docker_login()
    fmt = "'{{index .RepoDigests 0}}'"
    local_digest = get_digest(read(`docker inspect --format=$fmt $name`, String))
    remote_digest = get_digest(read(`docker buildx imagetools inspect $name`, String))
    local_digest == remote_digest
end

function update_local_server(task)
    name = docker_login()
    if task == "pull"
        run(`docker pull $name`)
    elseif task == "run"
        Threads.@spawn @handle_errors run(`docker run --rm --runtime=nvidia --gpus=all --network=host --name inference-server $name`)
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
    try
        run(`docker stop inference-server`)
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
while true
    sleep(3600)
    if !is_image_up_to_date()
        update()
    end
end
