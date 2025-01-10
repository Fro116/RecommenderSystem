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
    copy("environment", app)
end

function build(basedir::String, name::String, tag::String)
    run(`docker build -t $name $basedir`)
    repo = read("environment/docker/repo.txt", String)
    run(`docker tag $name $repo/$name:$tag`)
    run(`docker push $repo/$name:$tag`)
    deploy = read("environment/docker/deploy.txt", String)
    deploy = replace(deploy, "{app}" => name, "{tag}" => tag, "{args}" => "")
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
build(basedir, "read", tag)
