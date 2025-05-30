function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function embed_py(basedir::String)
    app = "$basedir/embed_py"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Training/transformer.model.py", app)
    copy("notebooks/Finetune/embed.py", app)
    mediums = [0, 1]
    files = vcat(
        ["manga.csv", "anime.csv", "finetune_tag", "training_tag"],
        ["transformer.$modeltype.$m.finetune.pt" for modeltype in ["causal", "masked"] for m in mediums],
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
    # deploy vm
    cmds = [
        "zone=`gcloud compute instances list --filter name=embed-$bluegreen --format 'csv[no-heading](zone)'`",
        "gcloud compute instances start embed-$bluegreen --zone \$zone",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    # deploy cloudrun backup
    cmds = [
        "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
        "gcloud run deploy {app} --image={repo}/{app}:{tag} --region={region} --project={project} {args}",
        "gcloud beta run services update {app} --scaling=auto --region {region}",
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
    # the first deploy can fail if startup takes too long
    # future deploys succeed because the image is already pulled
    cmd = "$deploy || (sleep 10 && $deploy)"
    run(`sh -c $cmd`)
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
