import CodecZstd
import Glob
import MsgPack
import ProgressMeter
import ProgressMeter: @showprogress

include("../../julia_utils/hash.jl")
include("../../julia_utils/stdout.jl")
include("import_history.jl")
const datadir = "../../../data/import/lists"

function decompress(x::AbstractString)
    MsgPack.unpack(
        CodecZstd.transcode(CodecZstd.ZstdDecompressor, Vector{UInt8}(hex2bytes(x[3:end]))),
    )
end

function get_directories(db::AbstractString)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    dirs = [chop(x) for x in split(str) if endswith(x, "/")]
    latest = read(`rclone cat r2:rsys/database/$db/latest`, String)
    [x for x in dirs if x <= latest]
end

function import_list(datetag::AbstractString, name::AbstractString)
    cmds = [
        "cd $datadir",
        "rclone -Pv --retries=10 copyto r2:rsys/database/lists/$datetag/$name.zstd $name.zstd",
        "unzstd $name.zstd",
        "rm $name.zstd",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function import_lists(datetag::AbstractString)
    tags_to_import = sort(get_directories("lists"))
    idx = findfirst(==(datetag), tags_to_import)
    if idx == 1
        logtag("HISTORIES", "$datetag has no previous histories")
        open("$datadir/histories.csv", "w") do f
            write(f, "source,username,userid,data,db_refreshed_at\n")
        end
    else
        import_list(tags_to_import[idx-1], "histories.csv")
    end
    import_list(datetag, "lists.csv")
end

function partition(name::AbstractString)
    outdir = "$datadir/$name"
    mkpath(outdir)
    cols = Dict()
    p = ProgressMeter.ProgressUnknown()
    for line in eachline("$datadir/$name.csv")
        ProgressMeter.next!(p)
        row = split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        if isempty(cols)
            for (i, x) in Iterators.enumerate(row)
                cols[x] = i
            end
            continue
        end
        k = (row[cols["source"]], lowercase(row[cols["username"]]), row[cols["userid"]])
        key = string(shahash(k))
        part = key[end-1:end]
        if !ispath("$outdir/$part")
            mkpath("$outdir/$part")
        end
        open("$outdir/$part/$key", "w") do g
            write(g, line)
        end
    end
    ProgressMeter.finish!(p)
end

function advance_histories(datetag::AbstractString)
    function read_user(fn)
        line = read(fn, String)
        split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
    end

    # import new users
    @showprogress for outdir in readdir("$datadir/lists")
        if !ispath("$datadir/merged/$outdir")
            mkpath("$datadir/merged/$outdir")
        end
        Threads.@threads for fn in readdir("$datadir/lists/$outdir")
            source, username, userid, data, db_refreshed_at =
                read_user("$datadir/lists/$outdir/$fn")
            user = decompress(data)
            update = true
            if ispath("$datadir/histories/$outdir/$fn")
                h_source, h_username, h_userid, h_data, h_db_refreshed_at =
                    read_user("$datadir/histories/$outdir/$fn")
                @assert h_source == source &&
                        lowercase(h_username) == lowercase(username) &&
                        h_userid == userid
                if h_db_refreshed_at == db_refreshed_at
                    update = false
                else
                    hist = decompress(h_data)
                end
            else
                hist = nothing
            end
            if update
                user = histories.update_history(
                    hist,
                    user,
                    source,
                    parse(Float64, db_refreshed_at),
                    datetag,
                )
                write_data = "\\x" * bytes2hex(
                    CodecZstd.transcode(
                        CodecZstd.ZstdCompressor,
                        Vector{UInt8}(MsgPack.pack(user)),
                    ),
                )
            else
                write_data = h_data
            end
            data = [
                source,
                username,
                userid,
                write_data,
                db_refreshed_at,
            ]
            open("$datadir/merged/$outdir/$fn", "w") do f
                write(f, join(data, ","))
            end
        end
    end
    # copy existing users
    @showprogress for outdir in readdir("$datadir/histories")
        if !ispath("$datadir/merged/$outdir")
            mkpath("$datadir/merged/$outdir")
        end
        Threads.@threads for fn in readdir("$datadir/histories/$outdir")
            if !ispath("$datadir/lists/$outdir/$fn")
                cp("$datadir/histories/$outdir/$fn", "$datadir/merged/$outdir/$fn")
            end
        end
    end
end

function upload_histories(datetag::AbstractString)
    for x in ["lists", "histories"]
        rm("$datadir/$x.csv")
        rm("$datadir/x", recursive=true, force=true)
    end
    open("$datadir/new_histories.csv", "w") do f
        write(f, "source,username,userid,data,db_refreshed_at")
    end
    Threads.@threads for outdir in readdir("$datadir/merged")
        cmd = "find $datadir/merged/$outdir/ -maxdepth 1 -type f -print0 | xargs -0 awk '1' > $datadir/new_histories.$outdir.csv"
        run(`sh -c $cmd`)
    end
    cmds = [
        "cat $datadir/new_histories.csv $datadir/new_histories.*.csv > $datadir/merged.csv",
        "zstd $datadir/merged.csv -o $datadir/new_histories.csv.zstd",
        "rm $datadir/new_histories*.csv",
        "rclone -Pv --retries=10 copyto $datadir/new_histories.csv.zstd r2:rsys/database/lists/$datetag/histories.csv.zstd",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function save_histories(datetag::AbstractString)
    logtag("HISTORIES", "saving $datetag")
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    import_lists(datetag)
    partition("histories")
    partition("lists")
    advance_histories(datetag)
    upload_histories(datetag)
end

save_histories(ARGS[1])
