include("../notebooks/julia_utils/multithreading.jl")
include("../notebooks/julia_utils/scheduling.jl")
include("../notebooks/julia_utils/stdout.jl")

function update()
    workdir=pwd()
    run(`$workdir/../notebooks/Import/database.sh $workdir/../secrets`)
end

@scheduled "DATABASE" "05:00" @handle_errors update()
