const LAYER_3_URL = ARGS[1]

import CSV
import DataFrames
import Glob
import JSON3
import Random

include("../julia_utils/database.jl")
include("../julia_utils/hash.jl")
include("../julia_utils/http.jl")
include("../julia_utils/multithreading.jl")
include("../julia_utils/stdout.jl")

function get_media_df(source)
    with_db(:prioritize) do db
        query = """
            SELECT * FROM $(source)_media 
            """
        stmt = db_prepare(db, query)
        df = DataFrames.DataFrame(LibPQ.execute(stmt))
    end
end

function get_mal_urls()
    df = get_media_df("mal")
    urls = []
    for i = 1:DataFrames.nrow(df)
        json = df.pictures[i]
        if ismissing(json)
            continue
        end
        for img in JSON3.read(json)
            url = nothing
            for k in [:large, :medium]
                if k in keys(img)
                    url = img[k]
                    break
                end
            end
            if !isnothing(url)
                url = split(url, "?")[1]
                push!(urls, ("mal", df.medium[i], df.itemid[i], url))
            end
        end
    end
    DataFrames.DataFrame(urls, [:source, :medium, :itemid, :imageurl])
end

function get_anilist_urls()
    df = get_media_df("anilist")
    urls = []
    for i = 1:DataFrames.nrow(df)
        json = df.coverimage[i]
        if ismissing(json)
            continue
        end
        img = JSON3.read(json)
        url = nothing
        for k in [:extraLarge, :large, :medium]
            if k in keys(img)
                url = img[k]
                break
            end
        end
        if !isnothing(url)
            url = split(url, "?")[1]
            push!(urls, ("anilist", df.medium[i], df.itemid[i], url))
        end
    end
    DataFrames.DataFrame(urls, [:source, :medium, :itemid, :imageurl])
end

function get_kitsu_urls()
    df = get_media_df("kitsu")
    urls = []
    for i = 1:DataFrames.nrow(df)
        json = df.posterimage[i]
        if ismissing(json)
            continue
        end
        img = JSON3.read(json)
        url = nothing
        for k in [:original, :large, :medium]
            if k in keys(img)
                url = img[k]
                break
            end
        end
        if !isnothing(url)
            url = split(url, "?")[1]
            push!(urls, ("kitsu", df.medium[i], df.itemid[i], url))
        end
    end
    DataFrames.DataFrame(urls, [:source, :medium, :itemid, :imageurl])
end

function get_animeplanet_urls()
    df = get_media_df("animeplanet")
    urls = []
    for i = 1:DataFrames.nrow(df)
        url = df.image[i]
        if ismissing(url)
            continue
        end
        url = split(url, "?")[1]
        push!(urls, ("animeplanet", df.medium[i], df.itemid[i], url))
    end
    DataFrames.DataFrame(urls, [:source, :medium, :itemid, :imageurl])
end

function get_path(df, i)
    source = df.source[i]
    medium = df.medium[i]
    itemid = df.itemid[i]
    url = df.imageurl[i]
    stem = split(url, ".")[end]
    hash = shahash(url)
    "$hash.$stem"
end

function get_urls()
    df = reduce(
        vcat,
        Random.shuffle.([get_mal_urls(), get_anilist_urls(), get_kitsu_urls(), get_animeplanet_urls()]),
    )
    df[!, "filename"] .= ""
    for i = 1:DataFrames.nrow(df)
        df[i, "filename"] = get_path(df, i)
    end
    hasimage(x) = ispath("$datadir/$x")
    df[!, "saved"] = hasimage.(df.filename)
    df
end

function save_image(source, url, fn)
    d = Dict("url" => url)
    endpoint = Dict(
        "mal" => "mal_image",
        "anilist" => "mal_image",
        "kitsu" => "mal_image",
        "animeplanet" => "animeplanet_image",
    )[source]
    r = HTTP.post(
        "$LAYER_3_URL/$endpoint",
        encode(d, :msgpack)...,
        status_exception = false,
    )
    if HTTP.iserror(r)
        logerror("failed to fetch $url")
        return false
    end
    data = decode(r)
    open("$datadir/$fn~", "w") do f
        write(f, data["image"])
    end
    if ispath("$datadir/$fn") && read("$datadir/$fn") == read("$datadir/$fn~")
        rm("$datadir/$fn~")
    else
        mv("$datadir/$fn~", "$datadir/$fn", force = true)
    end
    true
end

write_df(df) = CSV.write("$datadir/../images.csv", sort(df))

function save_new_images(df)
    logtag("IMAGES", "collecting $(sum(.!df.saved)) new images")
    num_saved = 0
    for i in 1:DataFrames.nrow(df)
        if df.saved[i]
            continue
        end
        saved = save_image(df.source[i], df.imageurl[i], df.filename[i])
        df.saved[i] |= saved
        num_saved += saved
        if num_saved % 10000 == 0
            write_df(df)
        end
    end
end

function save_random_images(df, N)
    for i in Random.shuffle(1:DataFrames.nrow(df))[1:N]
        df.saved[i] |= save_image(df.source[i], df.imageurl[i], df.filename[i])
    end
end

function cleanup(df)
    fns = setdiff(Set(Glob.glob("$datadir/*")), Set(["$datadir/$x" for x in df.filename]))
    logtag("IMAGES", "removing $(length(fns)) stale images")
    for x in fns
        rm(x, force = true, recursive = true)
    end
end

function save_images()
    while true
        df = get_urls()
        write_df(df)
        save_new_images(df)
        save_random_images(df, 10000)
        cleanup(df)
    end
end

const datadir = "../../data/collect/images"
if !ispath(datadir)
    mkpath(datadir)
end
save_images()
