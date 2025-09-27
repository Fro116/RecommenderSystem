import JSON3

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

function compress_media_json(app, medium)
    fn = "$app/data/finetune/$medium.json"
    json = open(fn) do f
        JSON3.read(f)
    end
    json = Base.copy(json)
    for x in json
        for k in [:recommendations, :reviews, :embedding]
            delete!(x, k)
        end
    end
    open(fn, "w") do f
        write(f, JSON3.write(json))
    end
end

function compute(basedir::String)
    app = "$basedir/compute"
    if ispath(app)
        rm(app; recursive = true)
    end
    copy("notebooks/Inference/compute.jl", app)
    copy("notebooks/Inference/render.jl", app)
    copy("notebooks/Training/import_list.jl", app)
    copy("notebooks/Finetune/embed.jl", app)
    mediums = ["manga", "anime"]
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    files = vcat(
        ["$m.$stem" for m in mediums for stem in ["csv", "json"]],
        ["media_relations.$m.jld2" for m in [0, 1]],
        ["training_tag", "finetune_tag", "model.registry.jld2", "images.csv", "item_similarity.jld2"],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
    copy("notebooks/julia_utils", app)
    copy("secrets", app)
    compress_media_json.(app, mediums)
end

function build(basedir::String, name::String, tag::String)
    repo = read("secrets/gcp.docker.txt", String)
    run(`docker container run --rm --runtime=nvidia --gpus all -p 5000:8080 --name embed embed`, wait = false)
    run(`docker build --network host -t $name $basedir`)
    run(`docker container stop embed`)
    # run(`docker tag $name $repo/$name:$tag`)
    # run(`docker push $repo/$name:$tag`)
    # run(`docker system prune -af --filter until=24h`)
    # project = read("secrets/gcp.project.txt", String)
    # region = read("secrets/gcp.region.txt", String)
    # group = read("secrets/gcp.igname.txt", String)
    # template = read("secrets/gcp.igtemplate.txt", String)
    # t = Int(round(time()))
    # cmds = [
    #     "gcloud auth login --cred-file=secrets/gcp.auth.json --quiet",
    #     "gcloud beta compute instance-groups managed rolling-action start-update $group --project=$project --region=$region --type=proactive --max-unavailable=0 --min-ready=0 --minimal-action=replace --replacement-method=substitute --max-surge=3 --version=template=$template,name=0-$t",
    # ]
    # deploy = replace(join(cmds, " && "))
    # run(`sh -c $deploy`)
end

cd("../../..")
basedir = "data/package/compute"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Compute/app", basedir, force = true)
layer1(basedir)
layer2(basedir)
layer3(basedir)
database(basedir)
compute(basedir)
build(basedir, "compute", "latest")
