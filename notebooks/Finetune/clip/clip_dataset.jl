import CSV
import DataFrames
import HypothesisTests: BinomialTest, confint
import JLD2
import JSON3
import Random
import Memoize: @memoize
import ProgressMeter: @showprogress
import Statistics
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
    n = Int(round(max(k, n) + w * 0.01))
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
    W = JLD2.load("$datadir/../watches.$medium.jld2")["$medium.watches"]
    W += W'
    df[:, :watches] .= 0
    for i = 1:DataFrames.nrow(df)
        df.watches[i] = W[df.source_matchedid[i]+1, df.target_matchedid[i]+1]
    end
    df[:, :score] = smoothed_wilson_score.(df.count, df.watches, df.source_popularity)
    df
end

function get_similar_pairs(source::AbstractString, medium::Int)
    # TODO weight sources
    mal_weight = 1
    anilist_weight = 1
    animeplanet_weight = 1
    df = CSV.read("$datadir/$source/$(source)_media.csv", DataFrames.DataFrame, ntasks = 1)
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
            counts = Dict()
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
                for _ = 1:rec["rating"]
                    k = (string(df.itemid[i]), string(rec.itemid))
                    counts[k] = get(counts, k, 0) + 1
                    imputed_username = "@anilist.$(minimum(k)).$(maximum(k)).$(counts[k])"
                    push!(
                        records,
                        (
                            source,
                            medium,
                            imputed_username,
                            string(df.itemid[i]),
                            string(rec.itemid),
                            anilist_weight,
                        ),
                    )
                end
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

function random_sample_excluded(n::Int, k::Int, exclude::Set{Int})
    @assert k <= n - length(exclude)
    result_set = Set{Int}()
    while length(result_set) < k
        candidate = rand(0:n-1)
        if candidate ∉ exclude && candidate ∉ result_set
            push!(result_set, candidate)
        end
    end
    collect(result_set)
end

function get_datasplits(df, test_frac::Float64, medium::Int, batch_size::Int)
    pairs = Set()
    for x in eachrow(df)
        k1 = min(x.source_matchedid, x.target_matchedid)
        k2 = max(x.source_matchedid, x.target_matchedid)
        push!(pairs, (k1, k2))
    end
    test_pairs = Set([x for x in pairs if rand() < test_frac])
    test_pairs = union(test_pairs, Set([(k2, k1) for (k1, k2) in test_pairs]))
    test_mask =
        [(x.source_matchedid, x.target_matchedid) in test_pairs for x in eachrow(df)]
    test = df[test_mask, :]
    training = df[.!test_mask, :]
    validation = df
    positives = Dict()
    popularity = Dict()
    for x in eachrow(test)
        if x.source_matchedid ∉ keys(positives)
            positives[x.source_matchedid] = Set([x.source_matchedid])
            popularity[x.source_matchedid] = x.source_popularity
        end
        push!(positives[x.source_matchedid], x.target_matchedid)
    end
    test_counts = Dict(k => length(v)-1 for (k, v) in positives)
    for x in eachrow(training)
        if x.source_matchedid ∉ keys(positives)
            continue
        end
        push!(positives[x.source_matchedid], x.target_matchedid)
    end
    records = []
    @showprogress for (k, v) in positives
        bs = batch_size - test_counts[k]
        if num_items(medium) - length(v) < bs
            @info "item $k medium $medium has too few implicit negatives"
            v = Set([x.source_matchedid])
        end
        negatives = random_sample_excluded(num_items(medium), bs, v)
        for n in negatives
            push!(records, ("medium$medium", k, n, 0, popularity[k], 0, 0))
        end
    end
    negative_df = DataFrames.DataFrame(
        records,
        [
            :cliptype,
            :source_matchedid,
            :target_matchedid,
            :count,
            :source_popularity,
            :watches,
            :score,
        ],
    )
    test = vcat(test, negative_df)
    training, validation, test
end

function save_similar_pairs(test_frac::Float64, batch_size::Int)
    sources = ["mal", "anilist", "kitsu", "animeplanet"]
    training_dfs = []
    validation_dfs = []
    test_dfs = []
    for medium in [0, 1]
        sdf = reduce(vcat, [get_similar_pairs(s, medium) for s in sources])
        training, validation, test =
            get_datasplits(aggragate_by_matchedid(sdf, medium), test_frac, medium, batch_size)
        push!(training_dfs, training)
        push!(validation_dfs, validation)
        push!(test_dfs, test)
    end
    training_df = reduce(vcat, training_dfs)
    validation_df = reduce(vcat, validation_dfs)
    test_df = reduce(vcat, test_dfs)
    CSV.write("$datadir/training.similarpairs.csv", training_df)
    CSV.write("$datadir/validation.similarpairs.csv", validation_df)
    CSV.write("$datadir/test.similarpairs.csv", test_df)
end

function get_adaptations(medium::Int)
    df = CSV.read("$datadir/../media_relations.csv", DataFrames.DataFrame)
    filter!(
        x ->
            x.source_medium == medium &&
                x.target_medium != medium &&
                x.relation in ["adaptation", "source"],
        df,
    )
    df = DataFrames.DataFrame(
        cliptype = "adaptation$medium",
        source_matchedid = df.source_matchedid,
        target_matchedid = df.target_matchedid,
    )
    df = DataFrames.filter(x -> x.source_matchedid != 0 && x.target_matchedid != 0, df)
    df
end

function save_adaptations(test_frac::Float64)
    df = reduce(vcat, [get_adaptations.([0, 1])...])
    function reflect(x)
        cliptype, sourceid, targetid = x
        reflect_type = Dict("adaptation0" => "adaptation1", "adaptation1" => "adaptation0")
        (reflect_type[cliptype], targetid, sourceid)
    end
    ids = Set()
    for i = 1:DataFrames.nrow(df)
        k = (df.cliptype[i], df.source_matchedid[i], df.target_matchedid[i])
        push!(ids, min(k, reflect(k)))
    end
    test_ids = Set(x for x in ids if rand() < test_frac)
    test_ids = union(test_ids, Set(reflect.(test_ids)))
    test_mask = [
        (x.cliptype, x.source_matchedid, x.target_matchedid) in test_ids for
        x in eachrow(df)
    ]
    training_df = df[.!test_mask, :]
    test_df = df[test_mask, :]
    CSV.write("$datadir/training.adaptations.csv", training_df)
    CSV.write("$datadir/test.adaptations.csv", test_df)
end

download_media()
save_similar_pairs(0.01, 2048)
save_adaptations(0.01)