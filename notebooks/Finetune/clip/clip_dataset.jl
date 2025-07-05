import CSV
import DataFrames
import JSON3
import Random
const datadir = "../../../data/finetune/clip"

function download_media(source::String)
    mkpath("$datadir/$source")
    retrieval = "rclone --retries=10 copyto r2:rsys/database/collect"
    cmd = "$retrieval/latest $datadir/$source/latest"
    run(`sh -c $cmd`)
    tag = read("$datadir/$source/latest", String)
    if source == "external"
        tables = ["dependencies"]
    else
        tables = ["media", "media_relations"]
    end
    for t in tables
        cmd = "$retrieval/$tag/$(source)_$(t).zstd $datadir/$source/$(source)_$(t).csv.zstd"
        run(`sh -c $cmd`)
        run(`unzstd -f $datadir/$source/$(source)_$(t).csv.zstd`)
        rm("$datadir/$source/$(source)_$(t).csv.zstd")
    end
end

function download_media()
    rm("$datadir", recursive = true, force = true)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    for s in sources
        download_media(s)
    end
end

function get_counts(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/../$m.csv", DataFrames.DataFrame, ntasks=1)
    df = DataFrames.combine(DataFrames.groupby(df, [:source, :matchedid])) do subdf
        subdf[argmax(subdf.count), :]
    end
    DataFrames.combine(DataFrames.groupby(df, [:matchedid]), :count => sum => :count)
end

function get_similar_pairs(source::AbstractString, medium::Int)
    # TODO weight sources
    mal_weight = 1
    anilist_weight = 1
    animeplanet_weight = 1
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks=1)
    records = []
    for i = 1:DataFrames.nrow(df)
        if source == "mal"
            m = Dict(0 => "manga", 1 => "anime")[medium]
            if ismissing(df.userrec[i]) || df.medium[i] != m
                continue
            end
            r = JSON3.read(df.userrec[i])
            for rec in r
                push!(
                    records,
                    (
                        source,
                        medium,
                        string(df.itemid[i]),
                        string(rec.itemid),
                        mal_weight,
                    ),
                )
            end
        elseif source == "anilist"
            m = Dict(0 => "MANGA", 1 => "ANIME")[medium]
            if ismissing(df.recommendationspeek[i])
                continue
            end
            r = JSON3.read(df.recommendationspeek[i])
            for rec in r
                if rec["medium"] != m
                    continue
                end
                @assert df.medium[i] == lowercase(rec["medium"])
                push!(
                    records,
                    (
                        source,
                        medium,
                        string(df.itemid[i]),
                        string(rec.itemid),
                        rec["rating"] * anilist_weight,
                    ),
                )
            end
        elseif source == "kitsu"
        elseif source == "animeplanet"
            m = Dict(0 => "manga", 1 => "anime")[medium]
            if ismissing(df.recommendations[i]) || df.medium[i] != m
                continue
            end
            r = JSON3.read(df.recommendations[i])
            for rec in r
                push!(
                    records,
                    (
                        source,
                        medium,
                        string(df.itemid[i]),
                        string(rec.itemid),
                        animeplanet_weight,
                    ),
                )
            end
        else
            @assert false
        end
    end
    if isempty(records)
        return DataFrames.DataFrame()
    end
    ret =
        DataFrames.DataFrame(records, ["source", "medium", "sourceid", "targetid", "count"])
    ret = DataFrames.combine(DataFrames.groupby(ret, [:source, :medium, :sourceid, :targetid]), :count => sum => :count)
    ret[:, :cliptype] .= "medium$(medium)"
    ret
end

function get_similar_pairs(medium::Int)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    df = reduce(vcat, [get_similar_pairs(s, medium) for s in sources])
    m = Dict(0 => "manga", 1 => "anime")[medium]
    group_df = CSV.read(
        "$datadir/../$m.csv",
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
        ntasks=1,
    )
    group_df = group_df[:, [:source, :itemid, :medium, :matchedid]]
    df = DataFrames.innerjoin(df, group_df, on = [:source, :medium, :sourceid => :itemid])
    df[:, :source_matchedid] = df.matchedid
    df = DataFrames.select!(df, DataFrames.Not(:matchedid))
    df = DataFrames.innerjoin(df, group_df, on = [:source, :medium, :targetid => :itemid])
    df[:, :target_matchedid] = df.matchedid
    df = DataFrames.select!(df, DataFrames.Not(:matchedid))
    df = DataFrames.select!(df, DataFrames.Not([:sourceid, :targetid]))
    df = DataFrames.combine(
        DataFrames.groupby(df, [:source, :medium, :source_matchedid, :target_matchedid]),
    ) do subdf
        subdf[argmax(subdf.count), :]
    end
    df = DataFrames.combine(
        DataFrames.groupby(df, [:cliptype, :source_matchedid, :target_matchedid]),
        :count => sum => :count,
    )
    df = DataFrames.filter(
        x -> x.source_matchedid != 0 && x.target_matchedid != 0 && x.count != 0,
        df,
    )
    counts = get_counts(medium)
    source_counts = DataFrames.rename(counts, :count => :source_popularity)
    df = DataFrames.innerjoin(df, source_counts, on= :source_matchedid => :matchedid)
    df
end

function get_adaptations(medium::Int)
    df = CSV.read("$datadir/../media_relations.csv", DataFrames.DataFrame)
    filter!(
        x -> x.source_medium == medium && x.target_medium != medium && x.relation in ["adaptation", "source"],
        df,
    )
    df = DataFrames.DataFrame(
        cliptype = "adaptation$medium",
        source_matchedid = df.source_matchedid,
        target_matchedid = df.target_matchedid,
        source_popularity = 1,
        count = 1,
    )
    df = DataFrames.filter(
        x -> x.source_matchedid != 0 && x.target_matchedid != 0 && x.count != 0,
        df,
    )
    df
end

function save_data()
    download_media()
    df = reduce(vcat, [get_similar_pairs.([0, 1])..., get_adaptations.([0, 1])...])
    display(
        DataFrames.combine(
            DataFrames.groupby(df, :cliptype),
            :source_matchedid => (x -> length(Set(x))) => "Items",
            DataFrames.nrow => "Pairs",
            :count => (c -> sum(abs.(c))) => "Votes",
        )
    )
    Random.shuffle!(df)
    test_frac = 0.1 # TODO decrease
    ids = Set()
    for i in 1:DataFrames.nrow(df)
        push!(ids, (df.cliptype[i], df.source_matchedid[i]))
        push!(ids, (df.cliptype[i], df.target_matchedid[i]))
    end
    test_ids = Set(x for x in ids if rand() < test_frac)
    mask = [(x.cliptype, x.source_matchedid) in test_ids || (x.cliptype, x.target_matchedid) in test_ids for x in eachrow(df)]
    test_df = df[mask, :]
    training_df = df[.!mask, :]
    CSV.write("$datadir/training.csv", training_df)
    CSV.write("$datadir/test.csv", test_df)
end

save_data()