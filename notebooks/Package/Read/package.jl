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

function build(basedir::String, name::String, tag::String, args::String)
    run(`docker build -t $name $basedir`)
    repo = read("secrets/gcp.docker.txt", String)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    run(`docker tag $name $repo/$name:$tag`)
    run(`docker push $repo/$name:$tag`)
    deploy = "gcloud auth login --cred-file=secrets/gcp.auth.json && gcloud run deploy {app} --image={repo}/{app}:{tag} --set-cloudsql-instances={project}:{region}:inference --execution-environment=gen2 --region={region} --project={project} {args} && gcloud auth revoke"
    deploy = replace(
        deploy,
        "{repo}" => repo,
        "{project}" => project,
        "{app}" => name,
        "{tag}" => tag,
        "{args}" => args,
    )
    run(`sh -c $deploy`)
end

cd("../../..")
basedir = "data/package/read"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Read/app", basedir, force = true)
layer4(basedir)
tag = Dates.format(Dates.today(), "yyyymmdd")
build(basedir, "read", tag, "")
