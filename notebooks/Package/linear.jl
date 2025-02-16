import CSV
import DataFrames
import Glob
import MsgPack
import Memoize: @memoize
import ProgressMeter: @showprogress
import Random
import SparseArrays

const datadir = "../../data/finetune"
const envdir = "../../environment"
const mediums = [0, 1]
const metrics = ["rating", "watch", "plantowatch", "drop"]
const planned_status = 3
const medium_map = Dict(0 => "manga", 1 => "anime")

@memoize function num_items(medium::Int)
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame).matchedid) + 1
end

function download()
    template = read("$envdir/database/download.txt", String)
    for m in mediums
        for metric in metrics
            for suffix in ["pt", "csv"]
                cmd = "$template/bagofwords.$m.$metric.finetune.$suffix $datadir/bagofwords.$m.$metric.finetune.$suffix"
                run(`sh -c $cmd`)
            end
        end
    end
end


download()
