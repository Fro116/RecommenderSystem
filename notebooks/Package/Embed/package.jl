import Dates

function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function layer4(basedir::String)
    app = "$basedir/layer4"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer4.jl", app)
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
end

function embed_py(basedir::String)
    app = "$basedir/embed_py"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Training/bagofwords.model.py", app)
    copy("notebooks/Finetune/embed.py", app)
    mediums = [0, 1]
    metrics = ["rating", "watch", "plantowatch", "drop"]
    files = vcat(
        ["manga.csv", "anime.csv"],
        ["latest", "training_tag"],
        ["bagofwords.$m.$metric.finetune.pt" for m in mediums for metric in metrics],
        ["baseline.$m.msgpack" for m in mediums],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
    copy("secrets", app)
end

function embed_jl(basedir::String)
    app = "$basedir/embed_jl"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Finetune/embed.jl", app)
    copy("notebooks/Training/import_list.jl", app)
    mediums = ["manga", "anime"]
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    files = vcat(
        ["$m.csv" for m in mediums],
        ["$(s)_$(m).csv" for s in sources for m in mediums],
        ["model.registry.jld2"],
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
    deploy = "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet && gcloud run deploy {app} --image={repo}/{app}:{tag} --set-cloudsql-instances={project}:{region}:inference --execution-environment=gen2 --region={region} --project={project} {args} && gcloud auth revoke"
    deploy = replace(
        deploy,
        "{repo}" => repo,
        "{project}" => project,
        "{region}" => region,
        "{app}" => name,
        "{tag}" => tag,
        "{args}" => args,
    )
    run(`sh -c $deploy`)
    run(`docker system prune -f`)
end

cd("../../..")
basedir = "data/package/embed"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Embed/app", basedir, force = true)
layer4(basedir)
embed_py(basedir)
embed_jl(basedir)
tag = Dates.format(Dates.today(), "yyyymmdd")
build(basedir, "embed", tag, "--cpu=8 --memory=16Gi")
