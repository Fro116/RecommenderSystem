import CSV
import DataFrames
import Glob
import JLD2
import Memoize: @memoize
import MsgPack
import ProgressMeter: @showprogress
import SparseArrays
include("history_tools.jl")
const datadir = "../../data/training"

@memoize function num_items(medium::Int)
    medium_map = Dict(0 => "manga", 1 => "anime")
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

function get_relations(source_medium::Int, target_medium::Int, relations)
    df = CSV.read("$datadir/media_relations.csv", DataFrames.DataFrame, ntasks=1)
    df = filter(
        x ->
            x.source_medium == source_medium &&
                x.target_medium == target_medium &&
                x.relation ∈ relations,
        df,
    )
    M = SparseArrays.sparse(
        convert.(Int32, df.source_matchedid .+ 1),
        convert.(Int32, df.target_matchedid .+ 1),
        fill(1.0f0, length(df.source_matchedid)),
        num_items(source_medium),
        num_items(target_medium),
    )
    M[M.>0] .= 1
    M
end

function transitive_closure(S)
    closure = convert.(Bool, S)
    for _ = 1:first(size(closure))
        new_closure = closure .| ((closure * closure) .> 0)
        if new_closure == closure
            break
        end
        closure = new_closure
    end
    convert.(eltype(S), closure)
end

function get_matrix(medium, relations; symmetric = false, transitive = false)
    S = get_relations(medium, medium, relations)
    if symmetric
        S = max.(S, S')
    end
    if transitive
        S = transitive_closure(S)
    end
    for i = 1:first(size(S))
        S[i, i] = 0
    end
    SparseArrays.dropzeros!(S)
    S
end

@memoize function get_media_df(medium::Int)
    medium_map = Dict(0 => "manga", 1 => "anime")
    m = medium_map[medium]
    CSV.read("$datadir/$m.csv", DataFrames.DataFrame; types = Dict(:startdate => String), ntasks=1)
end

function is_more_popular(medium::Int, cutoff, a1, a2)
    function get_popularity(itemid)
        df = get_media_df(medium)
        total = 0
        for i = 1:DataFrames.nrow(df)
            if df.matchedid[i] == itemid
                total += df.count[i]
            end
        end
        total
    end
    get_popularity(a1) > get_popularity(a2) * cutoff
end

function is_released_after(medium::Int, a1, a2)
    function get_startdate(itemid)
        df = get_media_df(medium)
        for i = 1:DataFrames.nrow(df)
            if df.matchedid[i] == itemid
                return df.startdate[i]
            end
        end
        missing
    end
    s1 = get_startdate(a1)
    s2 = get_startdate(a2)
    if ismissing(s1) || ismissing(s2)
        return false
    end
    f1 = split(s1, "-")
    f2 = split(s2, "-")
    for k = 1:min(length(f1), length(f2))
        if f1[k] > f2[k]
            return true
        end
        if f1[k] < f2[k]
            return false
        end
    end
    false
end

function save_dependencies(medium::Int)
    # M[i, j] = 1 if j should be watched before i
    relations = Set(["sequel", "prequel", "parent_story", "side_story"])
    M = get_matrix(medium, relations; symmetric = true)
    @showprogress for (a1, a2, _) in collect(zip(SparseArrays.findnz(M)...))
        watch_a1_before_a2 = (
            is_more_popular(medium, 0.9, a1 - 1, a2 - 1) &&
            !is_released_after(medium, a1 - 1, a2 - 1)
            # && !is_watched_after(medium, 0.6, a1 - 1, a2 - 1) # TODO test iswatchedafter
        )
        if !watch_a1_before_a2
            M[a2, a1] = 0
        end
    end
    SparseArrays.dropzeros!(M)
    M
end

function save_related(medium)
    # M[i, j] = 1 if i and j are in the same franchise
    relations = Set([
        "sequel",
        "prequel",
        "parent_story",
        "side_story",
        "alternative_version",
        "summary",
        "full_story",
        "adaptation",
        "alternative_setting",
    ])
    M = get_matrix(medium, relations; symmetric = true, transitive = true)
end

function save_recaps(medium)
    # M[i, j] = 1 if i and j are the same story
    relations = Set([
        "alternative_version",
        "summary",
        "full_story",
        "adaptation",
        "contains",
        "compilation",
    ])
    M = get_matrix(medium, relations; symmetric = true)
end

function save_adaptations(medium)
    # M[i, j] = 1 if i is an adaptation of j
    cross_medium = 1 - medium
    get_relations(medium, cross_medium, Set(["adaptation", "source"]))
end

@memoize function get_user_histories(medium)
    outdirs = Glob.glob("$datadir/users/training/*/")
    rets = []
    @showprogress for outdir in outdirs
        users = Glob.glob("$outdir/*.msgpack")
        histories = [Int32[] for _ = 1:length(users)]
        Threads.@threads for t = 1:length(users)
            user = MsgPack.unpack(read(users[t]))
            project_earliest!(user)
            for x in user["items"]
                if x["medium"] != medium
                    continue
                end
                push!(histories[t], x["matchedid"])
            end
        end
        push!(rets, histories)
    end
    reduce(vcat, rets)
end;

@memoize function get_item_to_histories(medium)
    histories = get_user_histories(medium)
    item_to_histories = Dict(a => Int64[] for a = 1:num_items(medium))
    @showprogress for i = 1:length(histories)
        for a in histories[i]
            if a == 0
                continue
            end
            push!(item_to_histories[a], i)
        end
    end
    Dict(k => Set(v) for (k, v) in item_to_histories)
end;

function is_watched_after(medium, cutoff, a1, a2)
    item_to_histories = get_item_to_histories(medium)
    idxs = collect(intersect(item_to_histories[a1], item_to_histories[a2]))
    if isempty(idxs)
        return false
    end
    counts = fill(false, length(idxs))
    Threads.@threads for i = 1:length(idxs)
        for a in histories[idxs[i]]
            if a == a2
                counts[i] = true
                break
            elseif a == a1
                break
            end
        end
    end
    sum(counts) / length(idxs) > cutoff
end

function save_data(m::Int)
    d = Dict()
    d["$m.dependencies"] = save_dependencies(m)
    d["$m.related"] = save_related(m)
    d["$m.recaps"] = save_recaps(m)
    d["$m.adaptations"] = save_adaptations(m)
    fn = "media_relations.$m.jld2"
    JLD2.save("$datadir/$fn", d)
    template = raw"tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => "$datadir/$fn",
        "{OUTPUT}" => fn,
    )
    run(`sh -c $cmd`)
end

save_data(parse(Int, ARGS[1]))