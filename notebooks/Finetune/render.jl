const registry = JLD2.load("$datadir/model.registry.jld2")
const relations = merge([JLD2.load("$datadir/media_relations.$m.jld2") for m in [0, 1]]...)
const planned_status = 3
const watching_status = 5

@memoize function num_items(medium::Integer)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/$m.csv", DataFrames.DataFrame, ntasks=1).matchedid) + 1
end

@memoize function get_images()
    df = CSV.read("$datadir/images.csv", DataFrames.DataFrame, ntasks=1)
    d = Dict()
    medium_map = Dict("manga" => 0, "anime" => 1)
    for i in 1:DataFrames.nrow(df)
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
        (1, "404.1.medium.webp", 1174, 1702),
        (1, "404.1.large.webp", 2348, 3404),
        (2, "404.2.medium.webp", 798, 1156),
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
                "height" => height
            )
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

function select_images(d)
    d = copy(d)
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
    df = CSV.read("$datadir/$m.csv", DataFrames.DataFrame; stringtype = String, ntasks=1)
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
        info[df.matchedid[i]] = Dict{String, Any}(
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

function compute(state, pagination)
    medium = state["medium"]
    project(user, name) = sum(
        (registry[kr]["weight"] * convert.(Float32, user[ke]) .+ registry[kr]["bias"]) *
        w for ((ke, kr), w) in registry[name]
    )
    x = zeros(Float32, num_items(medium))
    for user in state["users"]
        weight = user["weights"]
        x .+=
            project(user, "$medium.rating") *
            get(weight, "rating", 1) *
            get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.watch")) * (1 / log(4)) * get(weight, "watch", 1) * get(weight, "total", 1)
        x .+= logsoftmax(project(user, "$medium.status")) * 0 * get(weight, "status", 1) * get(weight, "total", 1)
    end
    # constraints
    function get_watched(user, m, status_filter = x -> x != planned_status)
        v = zeros(Float32, num_items(m))
        for (idx, status) in zip(user["$(m)_idx"], user["$(m)_status"])
            if status_filter(status)
                v[idx+1] = 1
            end
        end
        v
    end
    for user in state["users"]
        w = get_watched(user, medium)
        # seen
        for i in 1:length(x)
            if w[i] > 0
                x[i] = -Inf
            end
        end
        # adaptations
        v1 = relations["$(medium).adaptations"] * get_watched(user, 1 - medium)
        v2 = relations["$(medium).dependencies"] * w
        for i in 1:length(x)
            if v1[i] != 0 && v2[i] == 0
                x[i] = -Inf
            end
        end
        # recaps
        v = relations["$(medium).recaps"] * w
        for i in 1:length(x)
            if v[i] != 0
                x[i] = -Inf
            end
        end
        # dependencies
        v1 = relations["$(medium).dependencies"] * w
        v2 = relations["$(medium).dependencies"] * ones(Float32, num_items(medium))
        for i in 1:length(x)
            if v2[i] != 0 && v1[i] == 0
                x[i] = -Inf
            end
        end
        # sequels to currently watching / dropped
        v = relations["$(medium).dependencies"] * get_watched(user, medium, x -> x == watching_status || x > 0 && x < planned_status)
        for i in 1:length(x)
            if v[i] != 0
                x[i] = -Inf
            end
        end
        # TODO constrain by source
    end
    info = get_media_info(medium)
    ids = sortperm(x, rev = true) .- 1
    ids = [i for i in ids if i in keys(info) && x[i+1] > -Inf]
    total = length(ids)
    sidx = pagination["offset"] + 1
    eidx = pagination["offset"] + pagination["limit"]
    if sidx > total
        view = []
    else
        view = [select_images(info[i]) for i in ids[sidx:min(eidx, total)]]
    end
    view, total
end