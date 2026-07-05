import CSV
import DataFrames
import Images
import FileIO
import Glob
import Logging
import ProgressMeter: @showprogress
import Random
using ProgressMeter
include("../../julia_utils/hash.jl")
include("../../julia_utils/stdout.jl")

const basedir = "../../.."
const datadir = "$basedir/data/import/images"

qrun(x) = run(pipeline(x, stdout = devnull, stderr = devnull))

function import_data()
    qrun(`rclone --retries=10 copyto r2:rsys/database/collect/latest $datadir/latest`)
    tag = read("$datadir/latest", String)
    qrun(
        `rclone --retries=10 copyto r2:rsys/database/collect/$tag/images.tar $datadir/images.tar`,
    )
    cmds = ["cd $datadir", "tar xf images.tar", "rm images.tar"]
    cmd = join(cmds, " && ")
    qrun(`sh -c $cmd`)
    qrun(
        `rclone --retries=10 copyto r2:rsys/database/import/images.csv $datadir/srimages.csv`,
    )
end

function read_images_csv()
    df = CSV.read(
        "$datadir/images.csv",
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
    )
    filter(x -> x.saved, df)
end

read_srimages_csv() = CSV.read(
    "$datadir/srimages.csv",
    DataFrames.DataFrame,
    types = Dict("imageid" => String, "itemid" => String),
)

function get_new_hashes()
    df = read_images_csv()
    hashes = Dict{String,String}()
    for i = 1:DataFrames.nrow(df)
        imageid, _ = split(df.filename[i], ".")
        hashes[imageid] = df.imagehash[i]
    end
    hashes
end

function get_old_hashes()
    hashes = Dict{String,String}()
    if ispath("$datadir/srimages.csv")
        srdf = read_srimages_csv()
        for i = 1:DataFrames.nrow(srdf)
            hashes[srdf.imageid[i]] = srdf.imagehash[i]
        end
    end
    hashes
end

function get_new_images()
    new_hashes = get_new_hashes()
    old_hashes = get_old_hashes()
    df = read_images_csv()
    new_images = []
    for i = 1:DataFrames.nrow(df)
        imageid, _ = split(df.filename[i], ".")
        if get(old_hashes, imageid, nothing) != new_hashes[imageid]
            push!(new_images, df.filename[i])
        end
    end
    collect(Set(new_images))
end

function downsample(images)
    @showprogress Threads.@threads for name in images
        fn = "$datadir/images/$name"
        try
            run(`python downsample.py $fn 2000 2000`)
        catch e
            logerror("downsample: $fn error $e")
        end
    end
end

function super_resolution(src, dst)
    @assert isdir(src) && isdir(dst)
    abssrc = abspath(src)
    absdst = abspath(dst)
    models = Dict("large" => "noise_scale4x")
    for (tag, model) in models
        cmds = [
            "cd $basedir/../nunif",
            ". .venv/bin/activate",
            "python3 -m waifu2x.cli --style art_scan --method $model --noise-level 3 --batch-size 16 --gpu 0 -i $abssrc -o $absdst",
        ]
        cmd = join(cmds, " && ")
        for retry = 1:3
            try
                run(`sh -c $cmd`)
                break
            catch e
                logerror("error $e when running $cmd")
                sleep(30)
            end
        end
        Threads.@threads for x in Glob.glob("$dst/*.png")
            stem = join(split(x, ".")[1:end-1], ".")
            try
                qrun(`cwebp -q 80 $stem.png -o $stem.$tag.webp`)
            catch
                logtag("IMAGES", "failed command cwebp -q 80 $stem.png -o $stem.$tag.webp")
            end
        end
    end
end

function sr_images(images)
    tmpdir = "$datadir/tmp"
    rm("$datadir/srimages", recursive = true, force = true)
    mkpath("$datadir/srimages")
    @showprogress for batch in collect(Iterators.partition(images, 1024))
        rm(tmpdir, recursive = true, force = true)
        mkpath("$tmpdir/src")
        mkpath("$tmpdir/dst")
        for i = 1:length(batch)
            name, ext = split(batch[i], ".")
            cp("$datadir/images/$name.$ext", "$tmpdir/src/$i.$ext")
        end
        super_resolution("$tmpdir/src", "$tmpdir/dst")
        for i = 1:length(batch)
            name, ext = split(batch[i], ".")
            for x in ["large"]
                if !ispath("$tmpdir/dst/$i.$x.webp")
                    logtag(
                        "IMAGES",
                        "super_resolution of $name.$ext to $tmpdir/dst/$i.$x.webp failed",
                    )
                    continue
                end
                cp("$tmpdir/dst/$i.$x.webp", "$datadir/srimages/$name.$x.webp")
            end
        end
    end
end

function save_image_metadata()
    # existing images
    hashes = get_new_hashes()
    if ispath("$datadir/srimages.csv")
        srdf = read_srimages_csv()
        filter(x -> x.imageid ∉ keys(hashes), srdf)
    else
        srdf = DataFrames.DataFrame()
    end
    # newly added images
    df = read_images_csv()
    sizes = ["large"]
    rows = Any[nothing for _ = 1:DataFrames.nrow(df)*length(sizes)]
    Threads.@threads for i = 1:DataFrames.nrow(df)
        source = df.source[i]
        medium = df.medium[i]
        itemid = df.itemid[i]
        imagehash = df.imagehash[i]
        coverimage = df.coverimage[i]
        name, _ = split(df.filename[i], ".")
        for j = 1:length(sizes)
            basename = "$name.$(sizes[j]).webp"
            if !ispath("$datadir/srimages/$basename")
                continue
            end
            height, width = size(Images.load("$datadir/srimages/$basename"))
            rows[length(sizes)*(i-1)+j] = (
                source,
                medium,
                itemid,
                split(basename, ".")[1],
                basename,
                coverimage,
                height,
                width,
                imagehash,
            )
        end
    end
    rows = filter(x -> !isnothing(x), rows)
    if !isempty(rows)
        to_add = DataFrames.DataFrame(
            rows,
            [
                :source,
                :medium,
                :itemid,
                :imageid,
                :filename,
                :coverimage,
                :height,
                :width,
                :imagehash,
            ],
        )
        srdf = vcat(to_add, srdf)
    end
    CSV.write("$datadir/srimages.csv", sort(srdf))
end

function upload_images(to_add)
    if !isempty(to_add) && !isempty(readdir("$datadir/srimages"))
        logtag("IMAGES", "uploading images")
        run(`rclone --retries=10 copyto $datadir/srimages r2:cdn/images/cards`)
    end
    qrun(
        `rclone --retries=10 copyto $datadir/srimages.csv r2:rsys/database/import/images.csv`,
    )
    logtag("IMAGES", "finished uploading")
end

function encode_images()
    logtag("IMAGES", "importing data")
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    import_data()
    run(`python hash.py`)
    to_add = get_new_images()
    logtag("IMAGES", "adding $(length(to_add)) images")
    open("$datadir/to_add.txt", "w") do f
        foreach(x -> println(f, x), to_add)
    end
    downsample(to_add)
    sr_images(to_add)
    save_image_metadata()
    upload_images(to_add)
    rm(datadir, recursive = true, force = true)
end

encode_images()
