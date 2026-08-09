import Dates

function copy(file::String, dst::String)
    mkpath(joinpath(dst, dirname(file)))
    cp(file, joinpath(dst, file))
end

function runcmds(cmds)
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function build(basedir::String, name::String, tag::String, args::String)
    run(`docker build -t $name $basedir`)
    repo = read("secrets/gcp.docker.txt", String)
    project = read("secrets/gcp.project.txt", String)
    region = read("secrets/gcp.region.txt", String)
    runcmds(
        [
            "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
            "docker tag $name $repo/$name:$tag",
            "docker push $repo/$name:$tag",
            "gcloud run deploy $name --image=$repo/$name:$tag --region=$region --project=$project $args",
        ]
    )
    runcmds(["docker image prune -f", "docker builder prune -f --reserved-space=32GB"])
end

cd("../../..")
basedir = "data/package/client"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Client/app", basedir, force = true)
const tag = Dates.format(Dates.today(), "yyyymmdd")
build(basedir, "client", tag, "--min 1")
