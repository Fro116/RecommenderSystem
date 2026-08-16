import CSV
import DataFrames
import Dates
import JLD2
import Memoize: @memoize
import NNlib: gelu, logsoftmax, softmax
import SparseArrays
const datadir = "../../data/finetune";

const registry = JLD2.load("$datadir/model.registry.jld2")
const relations = merge([JLD2.load("$datadir/media_relations.$m.jld2") for m in [0, 1]]...)
const item_similarity = JLD2.load("$datadir/item_similarity.jld2")
const status_map = Dict(
    "none" => 0,
    "wont_watch" => 1,
    "dropped" => 2,
    "deleted" => 3,
    "on_hold" => 4,
    "planned" => 5,
    "currently_watching" => 6,
    "completed" => 7,
    "rewatching" => 8,
)

@memoize function num_items(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

@memoize function get_images()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame, ntasks = 1)
    d = Dict()
    medium_map = Dict("manga" => 0, "anime" => 1)
    for i = 1:DataFrames.nrow(df)
        key = (df.source[i], medium_map[df.medium[i]], string(df.itemid[i]))
        if key ∉ keys(d)
            d[key] = Dict()
        end
        imageid = df.imageid[i]
        if imageid ∉ keys(d[key])
            d[key][imageid] = []
        end
        val = Dict(
            "url" => "https://cdn.recs.moe/images/cards/$(df.filename[i])",
            "width" => df.width[i],
            "height" => df.height[i],
        )
        push!(d[key][imageid], val)
    end
    for k in keys(d)
        portrait_images = []
        landscape_images = []
        for x in collect(values(d[k]))
            if first(x)["height"] > first(x)["width"]
                push!(portrait_images, x)
            else
                push!(landscape_images, x)
            end
        end
        if !isempty(portrait_images)
            d[k] = portrait_images
        else
            d[k] = landscape_images
        end
    end
    d
end

@memoize function get_missing_images()
    error_images =
        [(1, "404.1.large.webp", 2348, 3404), (2, "404.2.large.webp", 1596, 2312)]
    groups = Dict()
    for x in error_images
        id, url, width, height = x
        if id ∉ keys(groups)
            groups[id] = []
        end
        push!(
            groups[id],
            Dict(
                "url" => "https://cdn.recs.moe/images/error/$url",
                "width" => width,
                "height" => height,
            ),
        )
    end
    collect(values(groups))
end

function get_item_url(source::String, medium::String, itemid::String)
    @assert medium in ["manga", "anime"]
    source_map = Dict(
        "mal" => "https://myanimelist.net",
        "anilist" => "https://anilist.co",
        "kitsu" => "https://kitsu.app",
        "animeplanet" => "https://anime-planet.com",
    )
    join([source_map[source], medium, itemid], "/")
end

function get_images(source, medium, itemid)
    images = get_images()
    key = (source, medium, string(itemid))
    if key ∉ keys(images)
        return nothing
    end
    images[key]
end

function render_card(d)
    d = copy(d)
    # choose a random image
    for k in ["image", "missing_image"]
        if isnothing(d["$(k)s"])
            d[k] = nothing
        else
            d[k] = rand(d["$(k)s"])
        end
        delete!(d, "$(k)s")
    end
    d
end

function is_unreleased(status, startdate)
    if !ismissing(startdate)
        today_parts = parse.(Int, split(Dates.format(Dates.now(), "yyyy-mm-dd"), "-"))
        start_parts = parse.(Int, split(startdate, "-"))
        while !isempty(start_parts) && start_parts[end] == 1
            pop!(start_parts)
        end
        matches = 0
        for (tp, sp) in zip(today_parts, start_parts)
            if tp > sp
                return false
            elseif tp < sp
                break
            else
                matches += 1
            end
        end
        if matches == length(today_parts)
            return false
        end
    end
    if !ismissing(status) && status in ["Upcoming", "TBA"]
        return true
    end
    false
end

@memoize function get_media_info(medium)
    info = Dict()
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = open("$datadir/$m.json") do f
        JSON3.read(f)
    end
    optint(x) = x != 0 ? x : missing
    optval(x) = isnothing(x) ? missing : x
    optdate(x) = isnothing(x) || isempty(x) ? missing : x
    function jsonlist(x)
        if isnothing(x) || isempty(x)
            return missing
        end
        valset = Set()
        vals = []
        for y in x
            if y ∉ valset
                push!(valset, y)
                push!(vals, y)
            end
        end
        join(vals, ", ")
    end
    function duration(x)
        if isnothing(x) || round(x) == 0
            return missing
        end
        d = []
        if x > 60
            hours = Int(div(x, 60))
            push!(d, "$hours hr.")
            x -= hours * 60
        end
        if x >= 1
            minutes = Int(floor(x))
            push!(d, "$minutes min.")
            x -= minutes
        end
        if isempty(d)
            seconds = Int(round(x * 60))
            push!(d, "$seconds sec.")
        end
        join(d, " ")
    end
    function season(x)
        if isnothing(x)
            return missing
        end
        season_str, year = split(x, "-")
        uppercasefirst(season_str) * " " * year
    end
    function english_title(title, engtitle)
        if isnothing(engtitle) || lowercase(title) == lowercase(engtitle)
            return missing
        end
        engtitle
    end
    for x in df
        key = first(x[:keys])
        info[x[:matchedid]] = Dict{String,Any}(
            "title" => x[:title],
            "english_title" => english_title(x[:title], x[:english_title]),
            "url" => get_item_url(key[2], key[1], key[3]),
            "type" => optval(x[:metadata][:mediatype]),
            "startdate" => optdate(x[:metadata][:dates][:startdate]),
            "enddate" => optdate(x[:metadata][:dates][:enddate]),
            "episodes" => optint(x[:metadata][:length][:episodes]),
            "duration" => duration(x[:metadata][:length][:duration]),
            "chapters" => optint(x[:metadata][:length][:chapters]),
            "volumes" => optint(x[:metadata][:length][:volumes]),
            "status" => optval(x[:metadata][:status]),
            "season" => season.(x[:metadata][:dates][:season]),
            "studios" => jsonlist(x[:metadata][:studios]),
            "source" => optval(x[:metadata][:source_material]),
            "genres" => jsonlist(x[:genres]),
            "synopsis" => first(x[:synopsis]),
            "images" => get_images(key[2], medium, key[3]),
            "missing_images" => get_missing_images(),
        )
    end
    max_ts = Dates.Date(read("$datadir/training_tag", String), Dates.DateFormat("yyyymmdd"))
    for (k, v) in collect(info)
        if is_unreleased(v["status"], v["startdate"])
            delete!(info, k)
        end
    end
    info
end

function retrieval(state)
    m = state["medium"]
    p = zeros(Float32, num_items(m))
    for a in state["items"]
        am = a["medium"]
        if m == am
            x = item_similarity["embeddings.$m"][:, a["matchedid"]+1]
        else
            x = item_similarity["embeddings.$am"][:, a["matchedid"]+1]
            x = item_similarity["crossproject.$am"] * x
        end
        p += item_similarity["embeddings.$m"]' * x
    end
    for u in state["users"]
        p += log.(compute_retrieval(registry, m, u["embeds"], 1:num_items(m)))
    end
    p[1] = -Inf # skip the default id for longtail items
    for u in state["users"]
        statuses = Dict(0 => Dict(), 1 => Dict())
        for x in u["user"]["items"]
            statuses[x["medium"]][x["matchedid"]+1] = x["status"]
        end
        # skip watched items
        for (x, s) in statuses[m]
            if s ∉ [status_map["deleted"], status_map["planned"]]
                p[x] = -Inf
            end
        end
        # skip adaptations
        watched = Dict(y => zeros(Float32, num_items(y)) for y in [0, 1])
        for y in [0, 1]
            for (x, s) in statuses[y]
                if s ∉ [status_map["deleted"], status_map["planned"]]
                    watched[y][x] = 1
                end
            end
        end
        v1 = relations["$(m).adaptations"] * watched[1-m]
        v2 = relations["$(m).dependencies"] * watched[m]
        for i = 1:length(p)
            if v1[i] != 0 && v2[i] == 0
                p[i] = -Inf
            end
        end
        # skip recaps
        v = relations["$(m).recaps"] * watched[m]
        for i = 1:length(p)
            if v[i] != 0
                p[i] = -Inf
            end
        end
        # skip works with missing dependencies
        v = zeros(Float32, num_items(m))
        for (x, s) in statuses[m]
            if s >= status_map["completed"]
                v[x] = 1
            end
        end
        v1 = relations["$(m).dependencies"] * v
        v2 = relations["$(m).dependencies"] * ones(Float32, num_items(m))
        for i = 1:length(p)
            if v2[i] != 0 && v1[i] == 0
                p[i] = -Inf
            end
        end
        # skip sequels to currently watching
        v = zeros(Float32, num_items(m))
        for (x, s) in statuses[m]
            if s in [
                status_map["currently_watching"],
                status_map["dropped"],
                status_map["wont_watch"],
            ]
                v[x] = 1
            end
        end
        v = relations["$(m).dependencies"] * v
        for i = 1:length(p)
            if v[i] != 0
                p[i] = -Inf
            end
        end
    end
    for a in state["items"]
        if m == a["medium"]
            # skip selected items
            p[a["matchedid"]+1] = -Inf
        end
    end
    info = get_media_info(m)
    ids = sortperm(p, rev = true)
    ids = [i for i in ids if (i - 1) in keys(info) && p[i] > -Inf]
    ids
end

function ranking(state, idxs, speedscope)
    m = state["medium"]
    ts = time()
    Threads.@threads for i = 1:length(state["users"])
        s = state["users"][i]
        source = s["source"]
        if "$m.ranking" in keys(s["embeds"]) && s["embeds"]["$m.ranking_idxs"] == idxs
            continue
        end
        u = copy(s["user"])
        u["embeds"] = Dict(k => s["embeds"][k] for k in ["$m.retrieval"])
        d_embed = query_model(u, m, idxs .- 1)
        if isnothing(d_embed)
            return HTTP.Response(500, []), false
        end
        d_embed["$m.ranking_idxs"] = idxs
        s["embeds"] = merge(s["embeds"], d_embed)
    end
    push!(speedscope, ("ranking_model", time()))
    score = zeros(Float32, length(idxs))
    for user in state["users"]
        u = user["embeds"]
        p = compute_retrieval(registry, m, u, idxs)
        r = compute_ranking(registry, m, u)
        score += log.(p) + r
    end
    push!(speedscope, ("ranking_sort", time()))
    score, true
end

function reranking!(state, idxs, r, partialk, speedscope)
    # setup
    medium = state["medium"]
    decay = Float32(state["penalties"]["decay"])
    ids = Dict(x => i for (i, x) in Iterators.enumerate(idxs))
    r = convert(Vector{Float32}, r)
    n = length(idxs)
    score = zeros(Float32, n)
    related = relations["$(medium).related"]::SparseArrays.SparseMatrixCSC{Float32,Int32}
    related_rows = SparseArrays.rowvals(related)
    related_vals = SparseArrays.nonzeros(related)
    # mmr penalty
    mmr_penalties = zeros(Float32, n)
    mmr_penalty = Float32(state["penalties"]["mmr_penalty"])
    embs = convert(Matrix{Float32}, item_similarity["embeddings.$medium"][:, idxs])
    pairwise_similarity = embs' * embs
    function apply_mmr_penalty!(mmr_penalties, id)
        similarity = view(pairwise_similarity, :, id)
        @. mmr_penalties = max(mmr_penalties * decay, similarity * mmr_penalty)
    end
    # same series penalty
    same_series_penalties = zeros(Float32, n)
    same_series_penalty = Float32(state["penalties"]["same_series_penalty"])
    function apply_same_series_penalty!(same_series_penalties, id)
        same_series_penalties .*= decay
        for k in SparseArrays.nzrange(related, idxs[id])
            i = related_rows[k]
            if related_vals[k] != 0 && haskey(ids, i)
                same_series_penalties[ids[i]] += same_series_penalty
            end
        end
    end
    # related penalty
    related_penalties = zeros(Float32, n)
    related_penalty = Float32(state["penalties"]["related_penalty"])
    related_ids = zeros(Float32, n)
    for u in state["users"]
        for x in u["user"]["items"]
            if x["medium"] != medium ||
               x["status"] in (status_map["deleted"], status_map["planned"])
                continue
            end
            for k in SparseArrays.nzrange(related, x["matchedid"] + 1)
                i = related_rows[k]
                if related_vals[k] != 0 && haskey(ids, i)
                    related_ids[ids[i]] = 1
                end
            end
        end
    end
    function apply_related_penalty!(related_penalties, id)
        related_penalties .*= decay
        if related_ids[id] != 0
            @. related_penalties += related_ids * related_penalty
        end
    end
    # ordering
    selected_ids = Int[]
    sizehint!(selected_ids, n)
    for _ in 1:min(partialk, n)
        @. score = r - mmr_penalties - same_series_penalties - related_penalties
        bestid = argmax(score)
        push!(selected_ids, bestid)
        r[bestid] = -Inf32
        apply_same_series_penalty!(same_series_penalties, bestid)
        apply_related_penalty!(related_penalties, bestid)
        apply_mmr_penalty!(mmr_penalties, bestid)
    end
    push!(speedscope, ("reranking", time()))
    idxs[selected_ids]
end

function render(state, pagination, speedscope)
    # add types
    for u in state["users"]
        for k in keys(u["embeds"])
            if k ∉ ["version", "0.ranking_idxs", "1.ranking_idxs"]
                u["embeds"][k] = Float32.(u["embeds"][k])
            end
        end
    end
    # retrieval
    idxs = retrieval(state)
    max_items_to_rank = 1024
    max_items_to_rank -= (max_items_to_rank % pagination["limit"])
    total = length(idxs)
    sidx = pagination["offset"] + 1
    eidx = pagination["offset"] + pagination["limit"]
    if sidx > length(idxs)
        view = []
        return (view, total), true
    end
    if eidx > total
        eidx = total
    end
    page = div(sidx - 1, max_items_to_rank, RoundDown)
    idxs = idxs[page*max_items_to_rank+1:(page+1)*max_items_to_rank]
    sidx -= page * max_items_to_rank
    eidx -= page * max_items_to_rank
    push!(speedscope, ("retrieval_sort", time()))
    # ranking
    r, ok = ranking(state, idxs, speedscope)
    if !ok
        return r, false
    end
    ids = reranking!(state, idxs, r, eidx, speedscope) .- 1
    info = get_media_info(state["medium"])
    view = [render_card(info[i]) for i in ids[sidx:eidx]]
    (view, total), true
end
