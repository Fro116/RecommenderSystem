import Dates

function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function layer1(basedir::String)
    app = "$basedir/layer1"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer1.py", app)
    copy("secrets", app)
end

function layer2(basedir::String)
    app = "$basedir/layer2"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer2.jl", app)
    copy("notebooks/Collect/entities.json", app)
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
end

function layer3(basedir::String)
    app = "$basedir/layer3"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer3.jl", app)
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
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
    run(`sh -c $deploy`)
    run(`docker system prune -af --filter until=24h`)
end

cd("../../..")
basedir = "data/package/database"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Database/app", basedir, force = true)
layer1(basedir)
layer2(basedir)
layer3(basedir)
database(basedir)
const tag = Dates.format(Dates.today(), "yyyymmdd")
build(basedir, "database", tag, "--set-cloudsql-instances={project}:{region}:inference --execution-environment=gen2 --cpu=1 --memory=2Gi --min=1 --max-instances=1")
