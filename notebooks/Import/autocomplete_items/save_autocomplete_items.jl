import CodecZstd
import CSV
import DataFrames
import JSON3
import MsgPack
import ProgressMeter: @showprogress

include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/autocomplete_items"
const mediums = ["manga", "anime"]
const sources = ["mal", "anilist", "kitsu", "animeplanet"]

qrun(x) = run(pipeline(x, stdout = devnull, stderr = devnull))

function download_data()
    logtag("SAVE_AUTOCOMPLETE_ITEMS", "downloading data")
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = "rclone --retries=10 copyto r2:rsys/database/import"
    files = vcat(
        ["$m.groups.csv" for m in mediums],
        ["$(s)_$(m).csv" for s in sources for m in mediums],
    )
    for fn in files
        cmd = "$retrieval/$fn $datadir/$fn"
        qrun(`sh -c $cmd`)
    end
end

function get_media(source, medium::String)
    fn = "$datadir/$(source)_$(medium).csv"
    df = CSV.read(fn, DataFrames.DataFrame, ntasks=1)
    parseint(x::Missing) = missing
    parseint(x::Real) = x
    parseint(x::AbstractString) = parse(Int, replace(x, "+" => ""))
    for c in [:episodes, :chapters, :volumes]
        df[!, c] = parseint.(df[:, c])
    end
    df[!, :source_material] = df[:, :source]
    df[!, :source] = fill(source, DataFrames.nrow(df))
    medium_map = Dict("manga" => 0, "anime" => 1)
    df[!, :medium] = fill(medium_map[medium], DataFrames.nrow(df))
    df[!, :itemid] = string.(df[:, :itemid])
    df = df[:, DataFrames.Not([:malid, :anilistid])]
    df
end

get_media(medium::String) = reduce(vcat, [get_media(s, medium) for s in sources])

function get_media_groups(medium::AbstractString)
    fn = "$datadir/$medium.groups.csv"
    groups = CSV.read(fn, DataFrames.DataFrame, types = Dict("itemid" => String), ntasks=1)
    media = get_media(medium)
    df = DataFrames.innerjoin(groups, media, on = [:source, :itemid])
    sort!(df, :count, rev = true)
    df[!, :distinctid] .= 0
    df[!, :matchedid] .= 0
    min_count = 100
    distinctid = 0
    groupmap = Dict()
    for i = 1:DataFrames.nrow(df)
        if df.count[i] < min_count
            df[i, :distinctid] = 0
            df[i, :matchedid] = get(groupmap, df[i, :groupid], 0)
        else
            distinctid += 1
            if df[i, :groupid] ∉ keys(groupmap)
                groupmap[df[i, :groupid]] = length(groupmap) + 1
            end
            df[i, :distinctid] = distinctid
            df[i, :matchedid] = groupmap[df[i, :groupid]]
        end
    end
    df
end

function get_media_groups(medium::Int)
    df = get_media_groups(Dict(0 => "manga", 1 => "anime")[medium])
    filter(x -> !(coalesce(x.status, nothing) in ["Upcoming", "TBA"]), df)
end

function get_title_records(medium::Int)
    df = get_media_groups(medium)
    seen = Set()
    records = []
    for i = 1:DataFrames.nrow(df)
        k = df.matchedid[i]
        if k == 0 || k ∈ seen
            continue
        end
        push!(seen, k)
        if !ismissing(df.title[i])
            r = (df.matchedid[i], df.title[i], 2)
            push!(records, r)
        end
        if !ismissing(df.english_title[i])
            r = (df.matchedid[i], df.english_title[i], 1)
            push!(records, r)
        end
        if !ismissing(df.alternative_titles[i])
            for t in JSON3.read(df.alternative_titles[i])
                r = (df.matchedid[i], t, 0)
                push!(records, r)
            end
        end
    end
    ret = DataFrames.DataFrame(records, [:matchedid, :title, :titletype])
    ret[:, :title] = lowercase.(ret.title)
    DataFrames.combine(
        DataFrames.groupby(ret, [:matchedid, :title]),
        :titletype => maximum => :titletype,
    )
end

function get_counts(medium::Int)
    df = get_media_groups(medium)
    df = DataFrames.combine(DataFrames.groupby(df, [:source, :matchedid])) do subdf
        subdf[argmax(subdf.count), :]
    end
    DataFrames.combine(DataFrames.groupby(df, [:matchedid]), :count => sum => :count)
end

function get_itemids(medium::Int)
    df = df = get_media_groups(medium)
    df = DataFrames.combine(DataFrames.groupby(df, [:matchedid])) do subdf
        subdf[argmax(subdf.count), :]
    end
    df[:, [:source, :itemid, :matchedid]]
end

function get_word_freq(titles::Vector{String})
    counts = Dict{String,Int}()
    total = 0
    for t in titles
        for w in split(t)
            counts[w] = get(counts, w, 0) + 1
            total += 1
        end
    end
    weights = Dict{String,Float64}()
    for (w, c) in counts
        rel = c / total
        weights[w] = rel
    end
    weights
end

function build_autocomplete_map(df)
    stop_words = Set(k for (k, v) in get_word_freq(df.title) if v > 1e-3 && length(k) <= 3)
    word_delims = Set(vcat([' ', '~'], [c for c = Char(0):Char(255) if ispunct(c)]))
    autocomplete = Dict{String,Vector{Tuple{String,String,Int,String,Float64}}}()
    @showprogress for row in eachrow(df)
        source = row.source
        itemid = row.itemid
        matchedid = row.matchedid
        title = row.title
        titletype = row.titletype
        count_bonus = log(row.count)
        idxs = collect(eachindex(title))
        word_starts = Set([
            first(idxs)
            [nextind(title, i) for i in idxs[1:end-1] if title[i] == ' ']
        ])
        for start in idxs
            pos = findfirst(==(start), idxs)
            start_bonus = start == idxs[1] ? 2.0 : 0.0
            midword_penalty = start in word_starts ? 0.0 : -100.0
            for j = pos:length(idxs)
                end_idx = idxs[j]
                substr = title[start:end_idx]
                title_bonus = (substr == title && titletype >= 1) ? 100.0 : 0.0
                nickname_bonus = (substr == title && titletype == 0) ? 1.0 : 0.0
                titletype_bonus = float(titletype)
                is_word_end =
                    (end_idx == last(idxs)) ||
                    (title[nextind(title, end_idx)] in word_delims)
                end_bonus = is_word_end ? 1.0 : 0.0
                stopword_penalty =
                    (substr in stop_words && start in word_starts && is_word_end) ? -2.0 :
                    0.0
                score =
                    count_bonus +
                    start_bonus +
                    title_bonus +
                    nickname_bonus +
                    titletype_bonus +
                    midword_penalty +
                    end_bonus +
                    stopword_penalty
                push!(
                    get!(autocomplete, substr, Vector{Tuple{String,String,Int,String,Float64}}()),
                    (source, itemid, matchedid, title, score),
                )
            end
        end
    end
    truncated_autocomplete = Dict{String,Vector{Tuple{String,String,String,Float64}}}()
    @showprogress for (key, vec) in autocomplete
        sort!(vec, by = x -> -x[end])
        seen = Set{Int}()
        newvec = Vector{Tuple{String,String,String,Float64}}()
        for (source, itemid, matchedid, title, score) in vec
            if !(matchedid in seen)
                push!(newvec, (source, itemid, title, score))
                push!(seen, matchedid)
                if length(newvec) == 10
                    break
                end
            end
        end
        truncated_autocomplete[key] = newvec
    end
    truncated_autocomplete
end

function text_encode(data)
    "\\x" * bytes2hex(
        CodecZstd.transcode(CodecZstd.ZstdCompressor, Vector{UInt8}(MsgPack.pack(data))),
    )
end

function get_autcompletes(medium::Int)
    df = get_title_records(medium)
    counts = get_counts(medium)
    itemids = get_itemids(medium)
    df = DataFrames.innerjoin(df, counts, on = :matchedid)
    df = DataFrames.innerjoin(df, itemids, on = :matchedid)
    ac = collect(build_autocomplete_map(df))
    records = Vector{Any}(undef, length(ac))
    Threads.@threads for i = 1:length(ac)
        k, v = ac[i]
        records[i] = (k, text_encode(v))
    end
    DataFrames.DataFrame(records, [:prefix, :data])
end

function save_autcompletes()
    logtag("SAVE_AUTOCOMPLETE_ITEMS", "saving autocompletes")
    dfs = []
    for m in [0, 1]
        df = get_autcompletes(m)
        df[:, :medium] .= m
        df = df[:, [:medium, :prefix, :data]]
        push!(dfs, df)
    end
    df = reduce(vcat, dfs)
    CSV.write("$datadir/item_autocomplete.csv", df, quotestrings=true)
end

function upload_autocompletes()
    cmds = [
        "cd ../../../data/import/autocomplete_items",
        "df=item_autocomplete",
        "db=autocomplete_items",
        raw"tail -n +2 $df.csv > $df.csv.headerless",
        raw"mv $df.csv.headerless $df.csv",
        raw"zstd $df.csv -o $df.csv.zstd",
        raw"rclone copyto -Pv $df.csv.zstd r2:rsys/database/import/$df.csv.zstd",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

download_data()
save_autcompletes()
logtag("SAVE_AUTOCOMPLETE_ITEMS", "uploading autocompletes")
upload_autocompletes()
rm(datadir, recursive = true, force = true)
