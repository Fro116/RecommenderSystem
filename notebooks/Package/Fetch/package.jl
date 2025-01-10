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
    copy("environment", app)
end

function layer2(basedir::String)
    app = "$basedir/layer2"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer2.jl", app)
    copy("notebooks/Collect/entities.json", app)
    copy("notebooks/julia_utils", app)
    copy("environment", app)
end

function layer3(basedir::String)
    app = "$basedir/layer3"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Collect/layer3.jl", app)
    copy("notebooks/julia_utils", app)
    copy("environment", app)
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
    deploy = replace(deploy, "{app}" => name, "{tag}" => tag, "{args}" => "--max-instances=1")
    run(`sh -c $deploy`)
end

cd("../../..")
basedir = "data/package/fetch"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Fetch/app", basedir, force = true)
layer1(basedir)
layer2(basedir)
layer3(basedir)
layer4(basedir)
tag = Dates.format(Dates.today(), "yyyymmdd")
build(basedir, "fetch", tag)
