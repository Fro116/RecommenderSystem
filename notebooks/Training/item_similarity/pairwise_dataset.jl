import CodecZstd
import CSV
import DataFrames
import H5Zblosc
import HDF5
import HypothesisTests: BinomialTest, confint
import JLD2
import JSON3
import Random
import Memoize: @memoize
import ProgressMeter: @showprogress
import Statistics
const datadir = "../../../data/training/item_similarity"

include("../../julia_utils/stdout.jl")

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
    sources = ["mal", "anilist", "kitsu", "animeplanet", "external"]
    for s in sources
        download_media(s)
    end
end

@memoize function num_items(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    maximum(CSV.read("$datadir/../$m.csv", DataFrames.DataFrame, ntasks = 1).matchedid) + 1
end

function get_counts(medium::Int)
    m = Dict(0 => "manga", 1 => "anime")[medium]
    df = CSV.read("$datadir/../$m.csv", DataFrames.DataFrame, ntasks = 1)
    df = DataFrames.combine(DataFrames.groupby(df, [:source, :matchedid])) do subdf
        subdf[argmax(subdf.count), :]
    end
    DataFrames.combine(DataFrames.groupby(df, [:matchedid]), :count => sum => :count)
end

function make_symmetric(df)
    records = Dict()
    for x in eachrow(df)
        k = (x.source, x.medium, x.username, sort([x.sourceid, x.targetid])...)
        if k ∉ keys(records)
            records[k] = x.count
        else
            records[k] = max(records[k], x.count)
        end
    end
    rows = []
    for (k, v) in records
        source, medium, username, sourceid, targetid = k
        push!(rows, (source, medium, username, sourceid, targetid, v))
        push!(rows, (source, medium, username, targetid, sourceid, v))
    end
    DataFrames.DataFrame(
        rows,
        ["source", "medium", "username", "sourceid", "targetid", "count"],
    )
end

function smoothed_wilson_score(k, n, w)
    k = min(n, k)
    n = n + Int(round(max((w - 2 * n), 0) * 0.05))
    interval = confint(BinomialTest(k, n), level = 1 - 0.05, method = :wald, tail = :both)
    lower_bound = interval[1]
    max(lower_bound, eps(Float32))
end

function aggragate_by_matchedid(df, medium)
    df = DataFrames.combine(
        DataFrames.groupby(df, [:cliptype, :source, :medium, :sourceid, :targetid]),
        :count => sum => :count,
    )
    m = Dict(0 => "manga", 1 => "anime")[medium]
    group_df = CSV.read(
        "$datadir/../$m.csv",
        DataFrames.DataFrame,
        types = Dict("itemid" => String),
        ntasks = 1,
    )
    group_df = group_df[:, [:source, :itemid, :medium, :matchedid]]
    df = DataFrames.innerjoin(df, group_df, on = [:source, :medium, :sourceid => :itemid])
    df[:, :source_matchedid] = df.matchedid
    df = DataFrames.select!(df, DataFrames.Not([:sourceid, :matchedid]))
    df = DataFrames.innerjoin(df, group_df, on = [:source, :medium, :targetid => :itemid])
    df[:, :target_matchedid] = df.matchedid
    df = DataFrames.select!(df, DataFrames.Not([:targetid, :matchedid]))
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
        x ->
            x.source_matchedid != 0 &&
                x.target_matchedid != 0 &&
                x.source_matchedid != x.target_matchedid &&
                x.count > 0,
        df,
    )
    counts = get_counts(medium)
    source_counts = DataFrames.rename(counts, :count => :source_popularity)
    df = DataFrames.innerjoin(df, source_counts, on = :source_matchedid => :matchedid)
    target_counts = DataFrames.rename(counts, :count => :target_popularity)
    df = DataFrames.innerjoin(df, target_counts, on = :target_matchedid => :matchedid)
    W = JLD2.load("$datadir/../watches.$medium.jld2")["$medium.watches"]
    W += W'
    df[:, :watches] .= 0
    for i = 1:DataFrames.nrow(df)
        df.watches[i] = W[df.source_matchedid[i]+1, df.target_matchedid[i]+1]
    end
    df[:, :score] = smoothed_wilson_score.(df.count, df.watches, df.source_popularity + df.target_popularity)
    df
end

function get_external(key::String)
    function unquote(x)
        if x[1] == '"' && x[end] == '"'
            return x[2:end-1]
        end
        x
    end
    # file contains fields that are too big for the CSV.jl parser
    lines = readlines("$datadir/external/external_dependencies.csv")
    headers = split(lines[1], ",")
    key_col = findfirst(==("key"), headers)
    value_col = findfirst(==("value"), headers)
    for line in lines
        fields = unquote.(split(line, ","))
        if fields[key_col] == key
            value = fields[value_col]
            bytes = hex2bytes(value[3:end])
            return String(CodecZstd.transcode(CodecZstd.ZstdDecompressor, bytes))
        end
    end
    @assert false
end

function get_similar_pairs(source::AbstractString, medium::Int)
    mal_weight = 1
    anilist_weight = 1
    animeplanet_weight = 1
    if source == "external"
        df =
            CSV.read(IOBuffer(get_external("item_similarity_manual")), DataFrames.DataFrame)
    else
        df = CSV.read(
            "$datadir/$source/$(source)_media.csv",
            DataFrames.DataFrame,
            ntasks = 1,
        )
    end
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
                        rec.username,
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
                        "",
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
                        rec.username,
                        string(df.itemid[i]),
                        string(rec.itemid),
                        animeplanet_weight,
                    ),
                )
            end
        elseif source == "external"
            if df.medium[i] != medium
                continue
            end
            @assert df.source1[i] == df.source2[i]
            push!(
                records,
                (
                    df.source1[i],
                    medium,
                    "",
                    string(df.itemid1[i]),
                    string(df.itemid2[i]),
                    df.count[i],
                ),
            )
        else
            @assert false
        end
    end
    if isempty(records)
        return DataFrames.DataFrame()
    end
    ret = DataFrames.DataFrame(
        records,
        ["source", "medium", "username", "sourceid", "targetid", "count"],
    )
    ret = make_symmetric(ret)
    ret[:, :cliptype] .= "medium$(medium)"
    ret
end

function save_similar_pairs()
    sources = ["mal", "anilist", "kitsu", "animeplanet", "external"]
    for medium in [0, 1]
        sdf = reduce(vcat, [get_similar_pairs(s, medium) for s in sources])
        all = aggragate_by_matchedid(sdf, medium)
        CSV.write("$datadir/pairs.$medium.csv", all)
        M = Matrix{Float32}(undef, num_items(medium), num_items(medium))
        for i in 1:size(M)[1]
            for j in i:size(M)[2]
                val = rand()
                M[i, j] = val
                M[j, i] = val
            end
        end
        M = M .< 0.02 # TODO gridsearch
        HDF5.h5open("$datadir/pairs.$medium.h5", "w") do file
            file["testmask", blosc = 3] = convert.(Int8, M)
        end
    end
end

logtag("ITEM_SIMILARITY", "downloading datasets")
download_media()
save_similar_pairs()
