include("../../julia_utils/stdout.jl")

const datetag = ARGS[1]
run(`julia save_lists.jl $datetag`)
run(`julia save_list_diffs.jl $datetag`)
run(`julia save_histories.jl $datetag $datetag`)
run(`julia archive.jl $datetag`)

logtag("SAVE_LISTS", "uploading $datetag")
const datadir = "../../../data/import/lists"
mkpath(datadir)
open("$datadir/latest", "w") do f
    write(f, datetag)
end
run(`rclone --retries=10 copyto $datadir/latest r2:rsys/database/lists/latest`)
