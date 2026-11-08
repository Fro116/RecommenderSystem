import UUIDs

include("../julia_utils/stdout.jl")
include("../julia_utils/multithreading.jl")
const datadir = "../../data/training"

function write_entrypoint()
    envvars = read("../../secrets/r2.auth.txt", String)
    script = read("entrypoint.gcp.sh", String)
    script = replace(script, "{{ENVVARS}}" => envvars)
    fn = "$datadir/entrypoint.gcp.sh"
    open(fn, "w") do f
        write(f, script)
    end
    fn
end

function get_b200_zones(region::String)
    try
        cmd = `gcloud compute machine-types list --filter="name=a4-highgpu-8g AND zone ~ ^$region" --format="value(zone)"`
        zones = unique(split(strip(read(cmd, String)), '\n', keepempty=false))
        return String.(zones)
    catch e
        logerror("get_b200_zones: $e")
        return ["$region-a", "$region-b", "$region-c", "$region-f"]
    end
end

function get_instance_status(instance_name::String, zone::String)
    try
        cmd = `gcloud compute instances describe $instance_name --zone=$zone --format="value(status)"`
        status = strip(read(cmd, String))
        return isempty(status) ? "NOT_FOUND" : String(status)
    catch
        return "NOT_FOUND"
    end
end

function delete_instance(instance_name::String, zone::String)
    logtag("RUNGPU", "Deleting VM instance $instance_name in $zone...")
    try
        run(`gcloud compute instances delete $instance_name --zone=$zone --no-user-output-enabled --quiet`)
    catch e
        logerror("delete_instance: $e")
    end
end

function try_launch_job(
    instance_name::String,
    zone::String,
    duration_hours::Int,
    disk_size_gb::Int,
    startup_script_path::String,
    wait_duration::String
)
    logtag("RUNGPU", "trying to launch $instance_name in $zone")
    cmd = `gcloud compute instances create $instance_name
        --zone=$zone
        --machine-type=a4-highgpu-8g
        --provisioning-model=FLEX_START
        --request-valid-for-duration=$wait_duration
        --max-run-duration=$(duration_hours)h
        --instance-termination-action=DELETE
        --maintenance-policy=TERMINATE
        --image-family=ubuntu-pro-accel-2404-amd64-nvidia-595
        --image-project=ubuntu-os-accelerator-images
        --boot-disk-type=hyperdisk-balanced
        --boot-disk-size=$(disk_size_gb)GB
        --metadata-from-file=startup-script=$startup_script_path
        --no-user-output-enabled
        --quiet`
    try
        run(cmd)
        return true
    catch e
        if get_instance_status(instance_name, zone) != "NOT_FOUND"
            delete_instance(instance_name, zone)
        end
        return false
    end
end

function launch_job(
    region::String,
    duration_hours::Int,
    disk_size_gb::Int,
    max_queue_wait_seconds::Real,
)
    startup_script_path = write_entrypoint()
    zones = get_b200_zones(region)
    queue_start_ts = time()
    queue_deadline = queue_start_ts + max_queue_wait_seconds
    attempt = 0
    while time() < queue_deadline
        attempt += 1
        queued_zone = zones[mod1(attempt, length(zones))]
        remaining_sec = Int(floor(queue_deadline - time()))
        if remaining_sec < 90
            break
        end
        wait_sec = min(7200, remaining_sec)
        instance_name = "b200-vm-" * string(UUIDs.uuid4())[1:8]
        if try_launch_job(instance_name, queued_zone, duration_hours, disk_size_gb, startup_script_path, "$(wait_sec)s")
            return instance_name, queued_zone
        end
    end
    nothing
end

function run_gpu_job()
    run(`gcloud auth login --quiet --cred-file=../../secrets/gcp.auth.json`)
    region::String = String(strip(read("../../secrets/gcp.region.txt", String)))
    duration_hours::Int = 1 # TODO switch back to 35 after testing
    disk_size_gb::Int = 96
    max_queue_wait_seconds::Real = 3600 * 24 * 7
    res = launch_job(
        region,
        duration_hours,
        disk_size_gb,
        max_queue_wait_seconds,
    )
    if isnothing(res)
        logerror("run_gpu_job: failed to launch job after $(max_queue_wait_seconds)s")
        return false
    end
    instance_name, zone = res
    logtag("RUNGPU", "launched $instance_name in $zone")
    job_start_ts = time()
    run_deadline = job_start_ts + duration_hours * 3600 + 1800
    try
        while time() < run_deadline
            status = get_instance_status(instance_name, zone)
            if status in ["TERMINATED", "STOPPED", "STOPPING", "NOT_FOUND"]
                return true
            end
            sleep(600)
        end
        logerror("run_gpu_job: job is still running after $(max_queue_wait_seconds)s.")
        return false
    finally
        delete_instance(instance_name, zone)
    end
end

run_gpu_job()