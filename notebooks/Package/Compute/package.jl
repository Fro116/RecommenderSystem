function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function compute(basedir::String)
    app = "$basedir/compute"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Inference/compute.jl", app)
    copy("notebooks/Inference/render.jl", app)
    copy("notebooks/Training/import_list.jl", app)
    copy("notebooks/Finetune/embed.jl", app)
    mediums = ["manga", "anime"]
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    files = vcat(
        ["$m.csv" for m in mediums],
        ["media_relations.$m.jld2" for m in [0, 1]],
        ["training_tag", "finetune_tag", "model.registry.jld2", "images.csv", "clip.jld2"],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
end

function build(basedir::String, name::String, tag::String)
    repo = read("secrets/gcp.docker.txt", String)
    run(`docker container run --rm --runtime=nvidia --gpus all -p 5000:8080 --name embed embed`, wait = false)
    run(`docker build --network host -t $name $basedir`)
    run(`docker container stop embed`)
    run(`docker tag $name $repo/$name:$tag`)
    run(`docker push $repo/$name:$tag`)
    run(`docker system prune -af --filter until=24h`)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    group = read("secrets/gcp.igname.txt", String)
    template = read("secrets/gcp.igtemplate.txt", String)
    t = Int(round(time()))
    cmds = [
        "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
        "gcloud beta compute instance-groups managed rolling-action start-update $group --project=$project --region=$region --type=proactive --max-unavailable=0 --min-ready=0 --minimal-action=replace --replacement-method=substitute --max-surge=3 --version=template=$template,name=0-$t",
    ]
    deploy = replace(join(cmds, " && "))
    run(`sh -c $deploy`)
end

cd("../../..")
basedir = "data/package/compute"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Compute/app", basedir, force = true)
compute(basedir)
build(basedir, "compute", "latest")
