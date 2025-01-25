import CSV
import DataFrames
include("common.jl")

function get_media(source, medium)
    CSV.read("$datadir/$(source)_$(medium).csv", DataFrames.DataFrame, ntasks = 1)
end

function get_mediatype_map(source::String, medium::String)
    df = get_media(source, medium)
    Dict(df.itemid .=> df.mediatype)
end

function save_mapping(source1::String, source2::String, medium::String)
    mappings = Set()
    if source2 == "mal" && source1 != "mal"
        col = "malid"
    elseif source2 == "anilist" && source1 != "anilist"
        col = "anilistid"
    else
        col = nothing
    end
    media = get_media(source1, medium)
    mediatype1 = get_mediatype_map(source1, medium)
    mediatype2 = get_mediatype_map(source2, medium)
    novel_types = Set(["Light Novel", "Novel"])
    if !isnothing(col)
        for (uid, mid) in zip(media.itemid, media[:, col])
            if !ismissing(uid) && !ismissing(mid)
                if !ismissing(get(mediatype1, uid, missing)) &&
                   !ismissing(get(mediatype2, mid, missing)) &&
                   xor(mediatype1[uid] in novel_types, mediatype2[mid] in novel_types)
                    continue
                end
                push!(mappings, (uid, mid))
            end
        end
    end
    colnames = ["source1", "source2"]
    if isempty(mappings)
        df = DataFrames.DataFrame(Dict(x => [] for x in colnames))
    else
        df = DataFrames.DataFrame(mappings, colnames)
    end
    CSV.write("$datadir/ids/$medium.$source1.$source2.csv", df)
end

function save_matches()
    mkpath("$datadir/ids")
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    for m in ["manga", "anime"]
        for i = 1:length(sources)
            for j = i+1:length(sources)
                save_mapping(sources[j], sources[i], m)
            end
        end
    end
end

save_matches()