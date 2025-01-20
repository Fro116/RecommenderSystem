import CSV
import DataFrames
import Memoize: @memoize
const datadir = "../../../data/media"
include("common.jl")

function merge_animeplanet_manga()
    # merge animeplanet manga that are split into parts    
    source = "animeplanet"
    medium = "manga"
    colnames = ["edgetype", "medium", "source1", "itemid1", "source2", "itemid2"]
    records = []
    df = CSV.read("$datadir/$(source)_media_relations.csv", DataFrames.DataFrame)
    df = filter(x -> x.medium == medium && x.target_medium == medium, df)
    for (s, t) in zip(df.itemid, df.target_id)
        prefix = s * "-part-"
        if startswith(t, prefix) && all(isdigit, t[length(prefix)+1:end])
            push!(records, ("merge", medium, source, s, source, t))
        end
    end
    if isempty(records)
        return DataFrames.DataFrame(Dict(x => [] for x in colnames))
    else
        return DataFrames.DataFrame(records, colnames)
    end
end

@memoize function get_matches()
    fn = "$datadir/external/media_match_manual.csv"
    if ispath(fn)
        df = CSV.read(fn, DataFrames.DataFrame)
    else
        df = CSV.read(IOBuffer(get_external("media_match_manual")), DataFrames.DataFrame)
    end
    CSV.write(fn, df)
    vcat(df, merge_animeplanet_manga())
end

function save_mapping(source1::String, source2::String, medium::String, edgetype::String)
    matches = get_matches()
    matches = filter(
        x ->
            x.edgetype == edgetype &&
                x.medium == medium &&
                x.source1 == source1 &&
                x.source2 == source2,
        matches,
    )
    colnames = ["source1", "source2"]
    mappings = Set()
    for (k1, k2) in zip(matches.itemid1, matches.itemid2)
        push!(mappings, (k1, k2))
    end
    if isempty(mappings)
        df = DataFrames.DataFrame(Dict(x => [] for x in colnames))
    else
        df = DataFrames.DataFrame(mappings, colnames)
    end
    CSV.write("$datadir/$edgetype/$medium.$source1.$source2.csv", df)
end


function save_matches()
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    for edgetype in ["invalid", "valid", "overlapping"]
        mkpath("$datadir/$edgetype")
        for m in ["manga", "anime"]
            for i = 1:length(sources)
                for j = i+1:length(sources)
                    save_mapping(sources[j], sources[i], m, edgetype)
                end
            end
        end
    end
    for edgetype in ["merge"]
        mkpath("$datadir/$edgetype")
        for m in ["manga", "anime"]
            for s in sources
                save_mapping(s, s, m, edgetype)
            end
        end
    end
end


save_matches()
