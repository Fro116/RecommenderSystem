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
    [chop(x) for x in split(str) if endswith(x, "/")]
end

function import_list(datetag::AbstractString, name::AbstractString)
    cmds = [
        "cd $datadir",
        "rclone --retries=10 copyto r2:rsys/database/lists/$datetag/$name.zstd $name.zstd",
        "unzstd $name.zstd",
        "rm $name.zstd",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
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
        if !isempty(row[cols["userid"]])
            # match across name changes
            k = ("userid", row[cols["source"]], row[cols["userid"]])
        else
            k = ("username", row[cols["source"]], lowercase(row[cols["username"]]))
        end
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
    rm("$datadir/$name.csv")
end

function advance_histories(datetag::AbstractString)
    function read_user(fn)
        line = read(fn, String)
        split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
    end
    @showprogress for outdir in readdir("$datadir/lists")
        if !ispath("$datadir/histories/$outdir")
            mkpath("$datadir/histories/$outdir")
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
                        (lowercase(h_username) == lowercase(username) ||
                        h_userid == userid)
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
            open("$datadir/histories/$outdir/$fn", "w") do f
                write(f, join(data, ","))
            end
        end
    end
end

function upload_histories(datetag::AbstractString)
    open("$datadir/new_histories.header", "w") do f
        write(f, "source,username,userid,data,db_refreshed_at\n")
    end
    Threads.@threads for outdir in readdir("$datadir/histories")
        cmd = "find $datadir/histories/$outdir/ -maxdepth 1 -type f -print0 | xargs -0 awk '1' > $datadir/new_histories.$outdir.csv"
        run(`sh -c $cmd`)
    end
    cmds = [
        "cat $datadir/new_histories.header $datadir/new_histories.*.csv > $datadir/new_histories.csv",
        "zstd $datadir/new_histories.csv -o $datadir/new_histories.csv.zstd",
        "rm $datadir/new_histories.*.csv",
        "rclone --retries=10 copyto $datadir/new_histories.csv.zstd r2:rsys/database/lists/$datetag/histories.csv.zstd",
        "rclone --retries=10 copyto $datadir/new_histories.csv.zstd r2:rsys/database/import/user_histories.csv.zstd",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function save_histories(startdate::AbstractString, enddate::AbstractString)
    logtag("SAVE_HISTORIES", "saving histories from $startdate to $enddate")
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    list_tags = sort(get_directories("lists"))
    sidx = findfirst(==(startdate), list_tags)
    eidx = findfirst(==(enddate), list_tags)
    @assert !isnothing(sidx) && !isnothing(eidx)
    if sidx == 1
        logtag("HISTORIES", "$startdate has no previous histories")
        open("$datadir/histories.csv", "w") do f
            write(f, "source,username,userid,data,db_refreshed_at\n")
        end
    else
        import_list(list_tags[sidx-1], "histories.csv")
    end
    partition("histories")
    for i in sidx:eidx
        datetag = list_tags[i]
        logtag("HISTORIES", "importing $datetag")
        import_list(datetag, "add.csv")
        mv("$datadir/add.csv", "$datadir/lists.csv")
        partition("lists")
        advance_histories(datetag)
        rm("$datadir/lists", recursive = true, force = true)
    end
    upload_histories(enddate)
    rm(datadir, recursive = true, force = true)
end

save_histories(ARGS[1], ARGS[2])
