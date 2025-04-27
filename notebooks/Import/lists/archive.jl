function get_files(db::AbstractString)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    [x for x in split(str) if !endswith(x, "/")]
end

function get_directories(db::AbstractString)
    str = read(`rclone lsf r2:rsys/database/$db`, String)
    [chop(x) for x in split(str) if endswith(x, "/")]
end

function archive(datetag::AbstractString)
    tags = sort(get_directories("lists"))
    idx = findfirst(==(datetag), tags)
    @assert !isnothing(idx)
    num_recent_tags_to_keep = 7
    if idx <= num_recent_tags_to_keep
        logtag(
            "ARCHIVE",
            "nothing to archive. keeping most recent " *
            "$num_recent_tags_to_keep folders before $datetag",
        )
        return
    end
    archive_tag = tags[idx-num_recent_tags_to_keep]
    logtag("ARCHIVE", "archiving $archive_tag")
    files = get_files("lists/$archive_tag")
    println(files)
    for k in ["lists", "add", "delete", "histories"]
        @assert "$k.csv.zstd" in files
    end
    for k in ["lists", "histories"]
        logtag("ARCHIVE", "deleting r2:rsys/database/$archive_tag/$k.csv.zstd")
        run(`rclone --retries=10 delete r2:rsys/database/$archive_tag/$k.csv.zstd`)
    end
    logtag("ARCHIVE", "deleting r2:rsys/database/collect/$archive_tag")
    run(`rclone purge r2:rsys/database/collect/$archive_tag`)
end

archive(ARGS[1])
