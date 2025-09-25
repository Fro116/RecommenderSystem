import CSV
import DataFrames
import Glob
import JLD2
import Memoize: @memoize
import MsgPack
import ProgressMeter: @showprogress
import SparseArrays
const datadir = "../../data/training"

@memoize function num_items(medium::Int)
    medium_map = Dict(0 => "manga", 1 => "anime")
    m = medium_map[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

function get_media_details()
    d = Dict()
    for medium in ["manga", "anime"]
        df = CSV.read("$datadir/$medium.csv", DataFrames.DataFrame, ntasks=1)
        for i = 1:DataFrames.nrow(df)
            k = (df.medium[i], df.matchedid[i])
            d[k] = Dict("mediatype" => df.mediatype[i])
        end
    end
    d
end

@memoize function get_media_relations()
    df = CSV.read("$datadir/media_relations.csv", DataFrames.DataFrame, ntasks=1)
    details = get_media_details()
    manga_types = Set(["Manhwa", "Manhua", "Manga", "OEL", "Doujinshi", "One-shot"])
    novel_types = Set(["Light Novel", "Novel"])
    for i = 1:DataFrames.nrow(df)
        if df.relation[i] != "unknown"
            continue
        end
        m1 = df.source_medium[i]
        id1 = df.source_matchedid[i]
        m2 = df.target_medium[i]
        id2 = df.target_matchedid[i]
        if m1 != m2
            df.relation[i] = "adaptation"
            continue
        else
            d1 = details[(m1, id1)]["mediatype"]
            d2 = details[(m2, id2)]["mediatype"]
            if d1 in manga_types && d2 in novel_types ||
               d1 in novel_types && d2 in manga_types
                df.relation[i] = "adaptation"
                continue
            end
        end
    end
    df
end

function get_relations(source_medium::Int, target_medium::Int, relations)
    df = get_media_relations()
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
        source_to_count = Dict()
        for i = 1:DataFrames.nrow(df)
            if df.matchedid[i] == itemid
                if df.source[i] ∉ keys(source_to_count)
                    source_to_count[df.source[i]] = 0
                end
                source_to_count[df.source[i]] = max(source_to_count[df.source[i]], df.count[i])
            end
        end
        sum(values(source_to_count))
    end
    get_popularity(a1) > (get_popularity(a1) + get_popularity(a2)) * cutoff
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

function project_earliest(user, medium)
    watching_status = 6
    seen = Set()
    items = []
    for x in user["items"]
        if x["medium"] != medium || x["matchedid"] in seen
            continue
        end
        watched = (x["status"] == 0) || (x["status"] >= watching_status)
        if !watched
            continue
        end
        push!(seen, x["matchedid"])
        push!(items, x["matchedid"])
    end
    items
end

@memoize function get_watch_order(medium)
    watch_order = zeros(Int32, num_items(medium), num_items(medium))
    num_users = 0
    outdirs = Glob.glob("$datadir/users/training/*/")
    @showprogress for outdir in outdirs
        users = Glob.glob("$outdir/*.msgpack")
        histories = [Int32[] for _ = 1:length(users)]
        Threads.@threads for t = 1:length(users)
            user = MsgPack.unpack(read(users[t]))
            histories[t] = project_earliest(user, medium)
        end
        for h in histories
            if length(h) > 0
                num_users += 1
            end
            for i in 1:length(h)
                for j in i+1:length(h)
                     watch_order[h[i]+1, h[j]+1] += 1
                end
            end
        end
    end
    watch_order, num_users
end

function is_watched_before(medium, cutoff, a1, a2)
    M, _ = get_watch_order(medium)
    M[a1+1, a2+1] > cutoff * (M[a1+1, a2+1] + M[a2+1, a1+1])
end

function save_dependencies(medium::Int)
    # M[i, j] = 1 if j should be watched before i
    relations = ["sequel", "prequel", "parent_story", "side_story"]
    R = sum([get_matrix(medium, [x]; transitive = true) for x in relations])
    R = R + R'
    M = get_matrix(medium, [])
    @showprogress for (i, j, v) in collect(zip(SparseArrays.findnz(R)...))
        if v == 0
            continue
        end
        watch_j_before_i = is_more_popular(medium, 0.5, j - 1, i - 1) &&
            is_watched_before(medium, 0.5, j - 1, i - 1) &&
            # check is_released_after using a double negative to handle
            # cases where release dates are unknown
            !is_released_after(medium, j - 1, i - 1)
        if watch_j_before_i
            M[i, j] = 1
        end
    end
    # remove transitive edges
    @showprogress for (i, j, v) in collect(zip(SparseArrays.findnz(M)...))
        for k in 1:size(M)[1]
            if M[i, j] > 0 && M[i, k] > 0 && M[k, j] > 0
                M[i, j] = 0
            end
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
        "spin_off",
        "compilation",
        "contains",
        "other",
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
    get_relations(medium, cross_medium, Set(["adaptation", "source", "alternative_version", "parent_story", "side_story"]))
end

function save_relations(m::Int)
    d = Dict()
    d["$m.dependencies"] = save_dependencies(m)
    d["$m.related"] = save_related(m)
    d["$m.recaps"] = save_recaps(m)
    d["$m.adaptations"] = save_adaptations(m)
    fn = "media_relations.$m.jld2"
    JLD2.save("$datadir/$fn", d)
    tag = read("$datadir/list_tag", String)
    template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => "$datadir/$fn",
        "{OUTPUT}" => fn,
    )
    run(`sh -c $cmd`)
end

function save_watch_order(m::Int)
    d = Dict()
    M, num_users = get_watch_order(m)
    d["$m.watches"] = M
    d["$m.users"] = num_users
    fn = "watches.$m.jld2"
    JLD2.save("$datadir/$fn", d)
    tag = read("$datadir/list_tag", String)
    template = "rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    cmd = replace(
        template,
        "{INPUT}" => "$datadir/$fn",
        "{OUTPUT}" => fn,
    )
    run(`sh -c $cmd`)
end

save_relations(parse(Int, ARGS[1]))
save_watch_order(parse(Int, ARGS[1]))
