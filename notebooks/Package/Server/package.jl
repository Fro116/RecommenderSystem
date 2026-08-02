import JSON3

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
        ["transformer.masked.finetune.base.pt"],
        ["transformer.masked.$m.$t.finetune.lora.pt" for m in mediums for t in ["watch", "rating"]],
    )
    for f in files
        copy("data/finetune/$f", app)
    end
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
    fields_to_keep = [:matchedid, :keys, :title, :english_title, :genres, :synopsis, :metadata]
    json = Base.copy(json)
    for x in json
        for k in collect(keys(x))
            if k ∉ fields_to_keep
                delete!(x, k)
            end
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
    run(`docker build --network host -t $name $basedir`)
    username, password, project = split(read("secrets/docker.auth.txt", String), "\n")
    cmds = [
        "docker login -u $username -p '$password'",
        "docker tag $name $username/$project-$name:$tag",
        "docker push $username/$project-$name:$tag"
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    repo = read("secrets/gcp.docker.txt", String)
    cmds = [
        "gcloud auth login --quiet --cred-file=secrets/gcp.auth.json",
        "docker tag $name $repo/$name:$tag",
        "docker push $repo/$name:$tag",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    repo = read("secrets/aws.docker.txt", String)
    region = read("secrets/aws.region.txt", String)
    account = read("secrets/aws.account.txt", String)
    cmds = [
        "export AWS_SHARED_CREDENTIALS_FILE=secrets/aws.credentials",
        "aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com",
        "docker tag $name $repo/$name:$tag",
        "docker push $repo/$name:$tag",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    cmds = [
        "docker image prune -f",
        "docker builder prune -f --keep-storage=32GB",
    ]
    run(`sh -c $cmd`)
end

cd("../../..")
basedir = "data/package/server"
if ispath(basedir)
    rm(basedir; recursive = true)
end
mkpath(basedir)
cp("notebooks/Package/Server/app", basedir, force = true)
embed_py(basedir)
layer1(basedir)
layer2(basedir)
layer3(basedir)
database(basedir)
compute(basedir)
build(basedir, "server", "latest")
