import CSV
import DataFrames
import JLD2
import Memoize: @memoize
import NNlib: gelu, logsoftmax
import SparseArrays
const datadir = "../../data/finetune";

const registry = JLD2.load("$datadir/model.registry.jld2")
const relations = merge([JLD2.load("$datadir/media_relations.$m.jld2") for m in [0, 1]]...)
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

@memoize function num_items(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

@memoize function get_images()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame, ntasks = 1)
    d = Dict()
    medium_map = Dict("manga" => 0, "anime" => 1)
    for i = 1:DataFrames.nrow(df)
        if df.width[i] >= df.height[i]
            continue
        end
        key = (df.source[i], medium_map[df.medium[i]], df.itemid[i])
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
        d[k] = collect(values(d[k]))
    end
    d
end

@memoize function get_missing_images()
    error_images = [
        (1, "404.1.large.webp", 2348, 3404),
        (2, "404.2.large.webp", 1596, 2312),
    ]
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

function get_url(source, medium, itemid)
    medium_map = Dict(0 => "manga", 1 => "anime")
    source_map = Dict(
        "mal" => "https://myanimelist.net",
        "anilist" => "https://anilist.co",
        "kitsu" => "https://kitsu.app",
        "animeplanet" => "https://anime-planet.com",
    )
    join([source_map[source], medium_map[medium], itemid], "/")
end

function get_images(source, medium, itemid)
    images = get_images()
    key = (source, medium, itemid)
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

@memoize function get_media_info(medium)
    info = Dict()
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks = 1)
    optint(x) = x != 0 ? x : missing
    function jsonlist(x)
        if ismissing(x) || isempty(x)
            return missing
        end
        join(JSON3.read(x), ", ")
    end
    function duration(x)
        if ismissing(x) || round(x) == 0
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
        if ismissing(x)
            return missing
        end
        season_str, year = split(x, "-")
        uppercasefirst(season_str) * " " * year
    end
    function english_title(title, engtitle)
        if ismissing(engtitle) || lowercase(title) == lowercase(engtitle)
            return missing
        end
        engtitle
    end
    for i = 1:DataFrames.nrow(df)
        if df.matchedid[i] == 0 || df.matchedid[i] in keys(info)
            continue
        end
        info[df.matchedid[i]] = Dict{String,Any}(
            "title" => df.title[i],
            "english_title" => english_title(df.title[i], df.english_title[i]),
            "url" => get_url(df.source[i], medium, df.itemid[i]),
            "type" => df.mediatype[i],
            "startdate" => df.startdate[i],
            "enddate" => df.enddate[i],
            "episodes" => optint(df.episodes[i]),
            "duration" => duration(df.duration[i]),
            "chapters" => optint(df.chapters[i]),
            "volumes" => optint(df.volumes[i]),
            "status" => df.status[i],
            "season" => season.(df.season[i]),
            "studios" => jsonlist(df.studios[i]),
            "source" => df.source_material[i],
            "genres" => jsonlist(df.genres[i]),
            "synopsis" => df.synopsis[i],
            "images" => get_images(df.source[i], medium, df.itemid[i]),
            "missing_images" => get_missing_images(),
        )
    end
    info
end

function retrieval(state)
    m = state["medium"]
    p = zeros(Float32, num_items(m))
    for u in state["users"]
        p .+= logsoftmax(
            registry["transformer.$m.embedding"] * u["transformer.$m"] +
            registry["transformer.$m.watch.bias"],
        )[1:num_items(m)]
    end
    for u in state["users"]
        # skip the default id for longtail items
        p[1] = -Inf
        # skip watched items
        for (x, s) in u["status"][m]
            if s ∉ [status_map["deleted"], status_map["planned"]]
                p[x] = -Inf
            end
        end
        # skip adaptations
        watched = Dict(y => zeros(Float32, num_items(y)) for y in [0, 1])
        for y in [0, 1]
            for (x, s) in u["status"][y]
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
        for (x, s) in u["status"][m]
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
        for (x, s) in u["status"][m]
            if s in [status_map["currently_watching"], status_map["dropped"], status_map["wont_watch"]]
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
    info = get_media_info(m)
    ids = sortperm(p, rev = true)
    ids = [i for i in ids if (i - 1) in keys(info) && p[i] > -Inf]
    ids
end

function ranking(state, idxs)
    m = state["medium"]
    r = zeros(Float32, length(idxs))
    for u in state["users"]
        p_watch = logsoftmax(
            registry["transformer.$m.embedding"] * u["transformer.$m"] +
            registry["transformer.$m.watch.bias"],
        )[idxs]
        p_baseline =
            only(registry["baseline.$m.rating.weight"]) *
            only(u["baseline.$m.rating"]) .+
            registry["baseline.$m.rating.bias"][idxs]
        p_bagofwords =
            registry["bagofwords.$m.rating.weight"][idxs, :] *
            u["bagofwords.$m.rating"] +
            registry["bagofwords.$m.rating.bias"][idxs]
        p_transformer = let
            # TODO make faster
            a = registry["transformer.$m.embedding"][idxs, :]'
            u_emb = repeat(u["transformer.$m"], 1, length(idxs))
            h = vcat(u_emb, a)
            h =
                registry["transformer.$m.rating.weight.1"] * h .+
                registry["transformer.$m.rating.bias.1"]
            h = gelu(h)
            h' * registry["transformer.$m.rating.weight.2"] .+
            only(registry["transformer.$m.rating.bias.2"])
        end
        p_transformer_baseline = registry["baseline.$m.rating.bias"][idxs]
        p_rating =
            sum(registry["$m.rating.coefs"] .* [p_baseline, p_bagofwords, p_transformer, p_transformer_baseline])
        r += p_rating + p_watch / log(4)
    end
    r
end

function render(state, pagination)
    # add types
    float32_cols = reduce(vcat, [["baseline.$m.rating", "bagofwords.$m.rating", "transformer.$m"] for m in [0, 1]])
    for u in state["users"]
        for k in float32_cols
            u[k] = Float32.(u[k])
        end
        for (k, v) in u
            @assert typeof(v) != Vector{Any} "untyped vector $k"
        end
    end
    # retrieval
    idxs = retrieval(state)
    max_items_to_rank = 1000 # for performance, only rank N items at a time
    max_items_to_rank += (max_items_to_rank % pagination["limit"])
    total = length(idxs)
    sidx = pagination["offset"] + 1
    eidx = pagination["offset"] + pagination["limit"]
    if sidx > length(idxs)
        view = []
        return view, total
    end
    if eidx > total
        eidx = total
    end
    page = div(sidx - 1, max_items_to_rank, RoundDown)
    idxs = idxs[page*max_items_to_rank+1:(page+1)*max_items_to_rank]
    sidx -= page * max_items_to_rank
    eidx -= page * max_items_to_rank
    # ranking
    r = ranking(state, idxs)
    ids = idxs[sortperm(r, rev = true)] .- 1
    info = get_media_info(state["medium"])
    view = [render_card(info[i]) for i in ids[sidx:eidx]]
    view, total
end
