import CodecZstd
import Glob
import ProgressMeter
import ProgressMeter: @showprogress

include("../../julia_utils/hash.jl")
include("../../julia_utils/stdout.jl")
const datadir = "../../../data/import/lists"
const datetag = ARGS[1]

function get_directories(db::AbstractString)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    [chop(x) for x in split(str) if endswith(x, "/")]
end

function import_list(datetag::AbstractString, name::AbstractString)
    cmds = [
        "cd $datadir",
        "rclone -Pv --retries=10 copyto r2:rsys/database/lists/$datetag/lists.csv.zstd $name.zstd",
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
        logtag("ARCHIVE", "$datetag has no previous lists")
        open("$datadir/previous.csv", "w") do f
            write(f, "source,username,userid,data,db_refreshed_at\n")
        end
    else
        import_list(tags_to_import[idx-1], "previous.csv")
    end
    import_list(datetag, "current.csv")
end

function get_key(row, cols)
    k = (row[cols["source"]], lowercase(row[cols["username"]]), row[cols["userid"]])
    string(shahash(k))
end

function get_existing_rows(name::AbstractString)
    d = Dict()
    cols = Dict()
    p = ProgressMeter.ProgressUnknown()
    for line in eachline("$datadir/$name")
        ProgressMeter.next!(p)
        row = split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        if isempty(cols)
            for (i, x) in Iterators.enumerate(row)
                cols[x] = i
            end
            continue
        end
        key = get_key(row, cols)
        d[key] = parse(Float64, row[cols["db_refreshed_at"]])
    end
    ProgressMeter.finish!(p)
    d
end

function write_deleted_rows()
    rows = get_existing_rows("current.csv")
    cols = Dict()
    p = ProgressMeter.ProgressUnknown()
    f = open("$datadir/delete.csv", "w")
    write(f, "source,username,userid\n")
    for line in eachline("$datadir/previous.csv")
        ProgressMeter.next!(p)
        row = split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        if isempty(cols)
            for (i, x) in Iterators.enumerate(row)
                cols[x] = i
            end
            continue
        end
        key = get_key(row, cols)
        if key ∉ keys(rows)
            vals = (row[cols["source"]], row[cols["username"]], row[cols["userid"]])
            write(f, join(vals, ","))
            write(f, "\n")
        end
    end
    close(f)
    ProgressMeter.finish!(p)    
end

function write_added_rows()
    rows = get_existing_rows("previous.csv")
    cols = Dict()
    p = ProgressMeter.ProgressUnknown()
    f = open("$datadir/add.csv", "w")
    write(f, "source,username,userid,data,db_refreshed_at\n")
    for line in eachline("$datadir/current.csv")
        ProgressMeter.next!(p)
        row = split(line, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
        if isempty(cols)
            for (i, x) in Iterators.enumerate(row)
                cols[x] = i
            end
            continue
        end
        key = get_key(row, cols)
        v = parse(Float64, row[cols["db_refreshed_at"]])
        if get(rows, key, nothing) != v
            write(f, line)
            write(f, "\n")
        end
    end
    close(f)
    ProgressMeter.finish!(p)    
end

function upload_diffs(datetag::AbstractString)
    for name in ["add.csv", "delete.csv"]
        cmds = [
            "zstd $datadir/$name -o $datadir/$name.zstd",
            "rclone -Pv --retries=10 copyto $datadir/$name.zstd r2:rsys/database/lists/$datetag/$name.zstd",
        ]
        cmd = join(cmds, " && ")
        run(`sh -c $cmd`)
    end
end

function archive_lists(datetag::AbstractString)
    logtag("ARCHIVE", "saving $datetag")
    rm(datadir, recursive = true, force = true)
    mkpath(datadir)
    import_lists(datetag)
    write_deleted_rows()
    write_added_rows()
    upload_diffs(datetag)
end

archive_lists(datetag)
