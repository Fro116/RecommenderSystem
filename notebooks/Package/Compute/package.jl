function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function database(basedir::String)
    app = "$basedir/database"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Inference/database.jl", app)
    copy("notebooks/Import/lists/import_history.jl", app)
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
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
        ["finetune_tag", "bluegreen", "model.registry.jld2", "images.csv", "clip.jld2"],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
end

function build(basedir::String, name::String, tag::String, args::String)
    run(`docker build --network host -t $name $basedir`)
    repo = read("secrets/gcp.docker.txt", String)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    run(`docker tag $name $repo/$name:$tag`)
    run(`docker push $repo/$name:$tag`)
    cmds = [
        "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
        "gcloud run deploy {app} --image={repo}/{app}:{tag} --region={region} --project={project} $args",
        "gcloud beta run services update {app} --scaling=auto --region {region}",
        "gcloud run services update {app} --min 1 --region {region}",
    ]
    deploy = replace(
        join(cmds, " && "),
        "{repo}" => repo,
        "{project}" => project,
        "{region}" => region,
        "{app}" => name,
        "{tag}" => tag,
        "{args}" => args,
    )
    for attempt in 1:3
        try
            run(`sh -c $deploy`)
            break
        catch e
            println("attempt $attempt: error $e")
        end
    end
    run(`docker system prune -af --filter until=24h`)
end

cd("../../..")
basedir = "data/package/compute"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Compute/app", basedir, force = true)
database(basedir)
compute(basedir)
const tag = read("data/finetune/finetune_tag", String)
build(basedir, "compute", tag, "--set-cloudsql-instances={project}:{region}:inference --execution-environment=gen2 --cpu=2 --memory=8Gi")
