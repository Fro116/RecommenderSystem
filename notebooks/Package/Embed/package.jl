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

function build(basedir::String, name::String, tag::String)
    run(`docker build -t $name $basedir`)
    repo = read("secrets/gcp.docker.txt", String)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    run(`docker tag $name $repo/$name:$tag`)
    run(`docker push $repo/$name:$tag`)
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
build(basedir, "embed", "latest")
