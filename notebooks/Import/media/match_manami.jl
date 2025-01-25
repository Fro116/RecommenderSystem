import CSV
import DataFrames
import JSON3
include("common.jl")

function save_mapping(source1::String, source2::String, medium::String)
    function get_keys(urls, source)
        ret = Set()
        for x in urls
            if source == "mal" && occursin("myanimelist.net", x)
                push!(ret, parse(Int, split(x, "/")[end]))
            elseif source == "anilist" && occursin("anilist.co", x)
                push!(ret, parse(Int, split(x, "/")[end]))
            elseif source == "kitsu" && occursin("kitsu.app", x)
                push!(ret, parse(Int, split(x, "/")[end]))
            elseif source == "animeplanet" && occursin("anime-planet.com", x)
                push!(ret, split(x, "/")[end])
            end
        end
        ret
    end
    colnames = ["source1", "source2"]
    mappings = Set()
    if medium == "anime"
        json = JSON3.read(get_external("manami"))
        valid_keys1 = get_valid_ids(source1, medium)
        valid_keys2 = get_valid_ids(source2, medium)
        for x in json["data"]
            ks1 = get_keys(x["sources"], source1)
            ks2 = get_keys(x["sources"], source2)
            for k1 in ks1
                for k2 in ks2
                    if k1 in valid_keys1 && k2 in valid_keys2
                        push!(mappings, (k1, k2))
                    end
                end
            end
        end
    end
    if isempty(mappings)
        df = DataFrames.DataFrame(Dict(x => [] for x in colnames))
    else
        df = DataFrames.DataFrame(mappings, colnames)
    end
    CSV.write("$datadir/manami/$medium.$source1.$source2.csv", df)
end

function save_matches()
    mkpath("$datadir/manami")
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
