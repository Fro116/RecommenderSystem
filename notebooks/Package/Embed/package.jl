function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function embed_py(basedir::String)
    app = "$basedir/embed_py"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Training/bagofwords.model.py", app)
    copy("notebooks/Training/transformer.model.py", app)
    copy("notebooks/Finetune/embed.py", app)
    mediums = [0, 1]
    files = vcat(
        ["manga.csv", "anime.csv", "finetune_tag", "training_tag", "transformer.config"],
        ["baseline.$metric.$m.msgpack" for m in mediums for metric in ["rating"]],
        ["bagofwords.$m.$metric.finetune.pt" for m in mediums for metric in ["rating"]],
        ["transformer.$m.finetune.pt" for m in mediums],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
end

function build(basedir::String, name::String, tag::String, args::String)
    run(`docker build -t $name $basedir`)
    repo = read("secrets/gcp.docker.txt", String)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    run(`docker tag $name $repo/$name-$bluegreen:$tag`)
    run(`docker push $repo/$name-$bluegreen:$tag`)
    cmds = [
        "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
        "gcloud run deploy {app} --image={repo}/{app}:{tag} --region={region} --project={project} {args}",
        "gcloud beta run services update {app} --scaling=auto --region {region}",
        "gcloud run services update {app} --min 1 --region {region}",
    ]
    deploy = replace(
        join(cmds, " && "),
        "{repo}" => repo,
        "{project}" => project,
        "{region}" => region,
        "{app}" => "$name-$bluegreen",
        "{tag}" => tag,
        "{args}" => args,
    )
    run(`sh -c $deploy`)
    run(`docker system prune -af --filter until=24h`)
end

cd("../../..")
const basedir = "data/package/embed"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Embed/app", basedir, force = true)
embed_py(basedir)
const tag = read("data/finetune/finetune_tag", String)
const bluegreen = read("data/finetune/bluegreen", String)
build(basedir, "embed", tag, "--cpu=4 --memory=16Gi")
