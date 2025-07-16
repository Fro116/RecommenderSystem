import CSV
import DataFrames
import Images
import Glob
import Logging
import OpenCV
import ProgressMeter: @showprogress
import Random

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

function add_imagehashes()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame)
    df[!, "imagehash"] .= ""
    Threads.@threads for i = 1:DataFrames.nrow(df)
        if !df.saved[i]
            continue
        end
        fn = "$datadir/images/$(df.filename[i])"
        @assert ispath(fn)
        df.imagehash[i] = open(fn) do f
            stringhash(read(f))
        end
    end
    CSV.write("$datadir/images.csv", df)
end

function get_diffs()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame)
    hashes = Dict{String,String}()
    src_exts = Dict()
    src_images = Set()
    dst_images = Set()
    for i = 1:DataFrames.nrow(df)
        fn = "$datadir/images/$(df.filename[i])"
        if !df.saved[i]
            continue
        end
        name, ext = split(df.filename[i], ".")
        src_exts[name] = ext
        hashes[name] = df.imagehash[i]
        push!(src_images, name)
    end
    if ispath("$datadir/srimages.csv")
        srdf = CSV.read("$datadir/srimages.csv", DataFrames.DataFrame)
        for i = 1:DataFrames.nrow(srdf)
            name, _, _ = split(srdf.filename[i], ".")
            if srdf.imagehash[i] != get(hashes, name, nothing)
                logerror("mismatched image hash for $(srdf.source[i]) $(srdf.itemid[i]) $name")
                continue
            end
            push!(dst_images, name)
        end
    end
    matched = intersect(src_images, dst_images)
    to_add = ["$k.$(src_exts[k])" for k in setdiff(src_images, dst_images)]
    to_delete =
        ["$k.$m.webp" for k in setdiff(dst_images, src_images) for m in ["large"]]
    matched, to_add, to_delete
end

function downsample(images)
    @showprogress for name in images
        fn = "$datadir/images/$name"
        while true
            img = OpenCV.imread(fn)
            width, height = OpenCV.size(img)[end-1:end]
            should_downsample = height >= 2000 || width >= 2000
            if !should_downsample
                break
            end
            width = Int32(div(width, 2))
            height = Int32(div(height, 2))
            img = OpenCV.resize(img, OpenCV.Size(width, height))
            OpenCV.imwrite(fn, img)
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
        for retry in 1:3
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
    rm("$datadir/srimages", recursive=true, force=true)
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

function save_image_metadata(to_delete)
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame, types = Dict("itemid" => String))
    df = filter(x -> x.saved, df)
    sizes = ["large"]
    rows = Any[nothing for _ = 1:DataFrames.nrow(df)*length(sizes)]
    Threads.@threads for i = 1:DataFrames.nrow(df)
        source = df.source[i]
        medium = df.medium[i]
        itemid = df.itemid[i]
        imagehash = df.imagehash[i]
        name, _ = split(df.filename[i], ".")
        for j = 1:length(sizes)
            basename = "$name.$(sizes[j]).webp"
            if !ispath("$datadir/srimages/$basename")
                continue
            end
            height, width = size(Images.load("$datadir/srimages/$basename"))
            rows[length(sizes)*(i-1)+j] =
                (source, medium, itemid, name, basename, height, width, imagehash)
        end
    end
    rows = filter(x -> !isnothing(x), rows)
    srdf = CSV.read(
        "$datadir/srimages.csv",
        DataFrames.DataFrame,
        types = Dict("imageid" => String, "itemid" => String),
    )
    if !isempty(rows)
        srdf_to_add = DataFrames.DataFrame(
            rows,
            [:source, :medium, :itemid, :imageid, :filename, :height, :width, :imagehash],
        )
        srdf = vcat(srdf, srdf_to_add)
    end
    filter!(x -> x.filename ∉ to_delete, srdf)
    CSV.write("$datadir/srimages.csv", sort(srdf))
end

function upload_images(to_add, to_delete)
    if !isempty(to_add) && !isempty(readdir("$datadir/srimages"))
        logtag("IMAGES", "uploading images")
        run(`rclone --retries=10 copyto $datadir/srimages r2:cdn/images/cards`)
    end
    if !isempty(to_delete)
        open("$datadir/todelete", "w") do f
            for fn in to_delete
                write(f, "$fn\n")
            end
        end
        logtag("IMAGES", "deleting stale images")
        run(`rclone --retries=10 delete r2:cdn/images/cards --files-from=$datadir/todelete --no-traverse`)
    end
    qrun(
        `rclone --retries=10 copyto $datadir/srimages.csv r2:rsys/database/import/images.csv`,
    )
end

function encode_images()
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    import_data()
    add_imagehashes()
    matched, to_add, to_delete = get_diffs()
    logtag(
        "IMAGES",
        "keeping $(length(matched)) adding $(length(to_add)) and deleting $(length(to_delete))",
    )
    downsample(to_add)
    sr_images(to_add)
    save_image_metadata(to_delete)
    upload_images(to_add, to_delete)
end

encode_images()
