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

cd(joinpath(@__DIR__, "../../.."))
basedir = "data/inference/fetch"
if ispath(basedir)
    rm(basedir; recursive = true)
end

cp("notebooks/Inference/Fetch/app", basedir, force = true)
layer1(basedir)
layer2(basedir)
layer3(basedir)
layer4(basedir)
