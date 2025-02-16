import Dates
include("../julia_utils/multithreading.jl")
include("../julia_utils/scheduling.jl")
include("../julia_utils/stdout.jl")


function get_gpu_args()
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    envvars = readlines("../../environment/gpu/envvars.sh")
    script = [
        "wget https://github.com/Fro116/RecommenderSystem/raw/main/notebooks/Training/entrypoint.sh",
        "chmod +x entrypoint.sh",
        "./entrypoint.sh",
    ]
    entrypoint = join(vcat(envvars, script), " && ")
    image, entrypoint
end

function start_runpod(gpuname)
    image, entrypoint = get_gpu_args()
    podname = "prod_train"
    create = [
        "runpodctl create pod",
        "--name $podname",
        "--gpuType '$gpuname'",
        "--gpuCount 8",
        "--imageName '$image'",
        "--volumePath /data",
        "--volumeSize 128",
        "--secureCloud",
        "--args 'bash -c \"$entrypoint\"'",
    ]
    create = join(create, " ")
    cmds = [
        create,
        "sleep 300",
        "runpodctl get pod | grep -w $podname | grep -w RUNNING",
    ]
    cmd = join(cmds, " && ")
    try
        run(`sh -c $cmd`)
        logtag("RUN", "started runpod with command $cmd")
        return true
    catch
        return false
    end
end

function start_gpu()
    while true
        #gputypes = ["NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H200"]
        gputypes = ["NVIDIA H100 80GB HBM3", "NVIDIA H200"]
        for gpuname in gputypes
            success = start_runpod(gpuname)
            if success
                return
            end
        end
        logtag("RUN", "waiting for available gpus...")
        sleep(600)
    end
end

start_gpu()

# function start_sfcompute()
#     image, entrypoint = get_gpu_args()
#     template = replace(
#         read("entrypoint.yaml", String),
#         "{IMAGE}" => image,
#         "{ENTRYPOINT}" => entrypoint,
#     )
#     fn = "../../data/training/entrypoint.yaml"
#     open(fn, "w") do f
#         write(f, template)
#     end
#     now = Dates.now()
#     start_hour = Dates.hour(now)
#     runtime_mins = 150
#     requested_hours = 1
#     allocated_mins = 60 - Dates.minute(now)
#     while allocated_mins < runtime_mins
#         requested_hours += 1
#         allocated_mins += 60
#     end
#     cmds = [
#         "sf buy -s $start_hour:00 --duration $(requested_hours)hr -n 8 -p 3 -y",
#         "clustername=`sf clusters list | grep -w name | awk '{print \$2}'`",
#         "sf clusters users add --cluster \$clustername --user myuser",
#         "kubectl apply -f $fn"
#     ]
#     cmd = join(cmds, " && ")
#     print(cmd)
#     return
#     try
#         run(`sh -c $cmd`)
#         logtag("RUN", "started sfcompute with command $cmd")
#         return true
#     catch
#         return false
#     end
# end

# start_sfcompute()

# function pretrain()
#     run(`julia import_data.jl`)
#     run(`julia baseline.jl`)
#     run(`julia -t auto bagofwords.jl --pretrain`)
#     start_gpu()
#     cmd = replace(
#         read("../../environment/database/storage.txt", String),
#         "{INPUT}" => "../../data/training/latest",
#         "{OUTPUT}" =>  "latest",
#     )
#     run(`sh -c $cmd`)
#     cleanup = read("../../environment/database/cleanup.txt", String)
#     run(`sh -c $cleanup`)
# end

# function finetune()
# end

# function train()
#     if Dates.dayofmonth(Dates.now()) in [1, 15]
#         pretrain()
#     end
#     finetune()
# end

# @scheduled "TRAIN" "7:00" @handle_errors train()
