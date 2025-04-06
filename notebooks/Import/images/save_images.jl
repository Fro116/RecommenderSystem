import CSV
import DataFrames
import Images
import Glob
import ProgressMeter: @showprogress
import Random

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

function get_diffs()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame)
    src_exts = Dict()
    src_images = Set()
    dst_images = Set()
    for i = 1:DataFrames.nrow(df)
        fn = "$datadir/images/$(df.filename[i])"
        @assert df.saved[i] == ispath(fn)
        if !ispath(fn)
            continue
        end
        name, ext = split(df.filename[i], ".")
        src_exts[name] = ext
        push!(src_images, name)
    end
    if ispath("$datadir/srimages.csv")
        srdf = CSV.read("$datadir/srimages.csv", DataFrames.DataFrame)
        for i = 1:DataFrames.nrow(srdf)
            name, _, _ = split(srdf.filename[i], ".")
            push!(dst_images, name)
        end
    end
    matched = intersect(src_images, dst_images)
    to_add = ["$k.$(src_exts[k])" for k in setdiff(src_images, dst_images)]
    to_delete =
        ["$k.$m.webp" for k in setdiff(dst_images, src_images) for m in ["medium", "large"]]
    matched, to_add, to_delete
end

function super_resolution(src, dst)
    @assert isdir(src) && isdir(dst)
    abssrc = abspath(src)
    absdst = abspath(dst)
    models = Dict("medium" => "noise_scale2x", "large" => "noise_scale4x")
    for (tag, model) in models
        cmds = [
            "cd $basedir/../nunif",
            ". .venv/bin/activate",
            "python3 -m waifu2x.cli --style art_scan --method $model --noise-level 3 -i $abssrc -o $absdst",
        ]
        cmd = join(cmds, " && ")
        qrun(`sh -c $cmd`)
        Threads.@threads for x in Glob.glob("$dst/*.png")
            stem = join(split(x, ".")[1:end-1], ".")
            qrun(`cwebp -q 80 $stem.png -o $stem.$tag.webp`)
        end
    end
end

function sr_images(images)
    tmpdir = "$datadir/tmp"
    mkpath("$datadir/srimages")
    @showprogress for batch in collect(Iterators.partition(images, 1000))
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
            for x in ["medium", "large"]
                if !ispath("$tmpdir/dst/$i.$x.webp")
                    logtag(
                        "IMAGES",
                        "super_resolution of $name to $tmpdir/dst/$i.$x.webp failed",
                    )
                    continue
                end
                cp("$tmpdir/dst/$i.$x.webp", "$datadir/srimages/$name.$x.webp")
            end
        end
    end
end

function save_image_metadata(to_add, to_delete)
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame)
    df = filter(x -> x.saved, df)
    sizes = ["medium", "large"]
    rows = Any[nothing for _ = 1:DataFrames.nrow(df)*length(sizes)]
    Threads.@threads for i = 1:DataFrames.nrow(df)
        source = df.source[i]
        medium = df.medium[i]
        itemid = df.itemid[i]
        name, _ = split(df.filename[i], ".")
        for j = 1:length(sizes)
            basename = "$name.$(sizes[j]).webp"
            if !ispath("$datadir/srimages/$basename")
                continue
            end
            height, width = size(Images.load("$datadir/srimages/$basename"))
            rows[length(sizes)*(i-1)+j] =
                (source, medium, itemid, name, basename, height, width)
        end
    end
    rows = filter(x -> !isnothing(x), rows)
    srdf = CSV.read(
        "$datadir/srimages.csv",
        DataFrames.DataFrame,
        types = Dict("imageid" => String),
    )
    if !isempty(rows)
        srdf_to_add = DataFrames.DataFrame(
            rows,
            [:source, :medium, :itemid, :imageid, :filename, :height, :width],
        )
        srdf = vcat(srdf, srdf_to_add)
    end
    filter!(x -> x.filename ∉ to_delete, srdf)
    CSV.write("$datadir/srimages.csv", sort(srdf))
end

function upload_images(to_add, to_delete)
    qrun(`rclone --retries=10 copyto $datadir/srimages r2:cdn/images/cards`)
    @showprogress for fn in to_delete
        qrun(`rclone --retries=10 delete r2:cdn/images/cards/$fn`)
    end
    qrun(
        `rclone --retries=10 copyto $datadir/srimages.csv r2:rsys/database/import/images.csv`,
    )
end

function encode_images()
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    import_data()
    matched, to_add, to_delete = get_diffs()
    logtag(
        "IMAGES",
        "keeping $(length(matched)) adding $(length(to_add)) and deleting $(length(to_delete))",
    )
    sr_images(to_add)
    save_image_metadata(to_add, to_delete)
    upload_images(to_add, to_delete)
end

encode_images()
