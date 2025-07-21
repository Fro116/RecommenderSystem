import CSV
import CodecZstd
import DataFrames
import Glob
import MsgPack
import ProgressMeter: @showprogress

include("../../julia_utils/stdout.jl")
include("../../Training/import_list.jl")

const datadir = "../../../data/import/autocomplete"

qrun(x) = run(pipeline(x, stdout = devnull, stderr = devnull))

function import_data()
    logtag("SAVE_AUTOCOMPLETE", "importing data")
    rm(datadir, recursive=true, force=true)
    mkpath(datadir)
    qrun(`rclone --retries=10 copyto r2:rsys/database/lists/latest $datadir/latest`)
    tag = read("$datadir/latest", String)
    qrun(
        `rclone --retries=10 copyto r2:rsys/database/lists/$tag/lists.csv.zstd $datadir/lists.csv.zstd`,
    )
    cmds = ["cd $datadir", "unzstd lists.csv.zstd", "rm lists.csv.zstd"]
    cmd = join(cmds, " && ")
    qrun(`sh -c $cmd`)
    qrun(`mlr --csv split -n 1000000 --prefix $datadir/lists $datadir/lists.csv`)
    rm("$datadir/lists.csv")
end

function save_profiles()
    logtag("SAVE_AUTOCOMPLETE", "saving profiles")
    @showprogress for (idx, f) in Iterators.enumerate(Glob.glob("$datadir/lists_*.csv"))
        df = read_csv(f)
        records = Vector{Any}(undef, DataFrames.nrow(df))
        Threads.@threads for i = 1:length(records)
            ts = parse(Float64, df.db_refreshed_at[i])
            data = decompress(df.data[i])
            user = import_profile(df.source[i], data, ts)
            r = (user["source"], user["username"], ts, user["avatar"], user["last_online"], user["gender"], user["birthday"], user["created_at"])
            records[i] = r
        end
        df = DataFrames.DataFrame(records, [:source, :username, :accessed_at, :avatar, :last_online, :gender, :birthday, :created_at])
        CSV.write(
            "$datadir/profiles.$idx.csv",
            df,
            transform = (col, val) -> something(val, missing),
        )
    end
    cmds = [
        "mlr --csv cat $datadir/profiles.*.csv > $datadir/profiles.csv ",
        "rm $datadir/profiles.*.csv",
        "rm $datadir/lists*.csv",
    ]
    cmd = join(cmds, " && ")
    qrun(`sh -c $cmd`)
end

mutable struct TrieNode{V}
    children::Dict{Char,TrieNode{V}}
    is_end::Bool
    metadata::Dict{String,V}
    function TrieNode{V}() where {V}
        new(Dict{Char,TrieNode{V}}(), false, Dict{String,V}())
    end
end

mutable struct AutoComplete{V}
    root::TrieNode{V}
end

function AutoComplete(user_dict::Dict{String,Dict{String,V}}) where {V}
    root = TrieNode{V}()
    for (username, meta) in user_dict
        insert!(root, username, meta)
    end
    AutoComplete{V}(root)
end

function insert!(node::TrieNode{V}, username::String, meta::Dict{String,V}) where {V}
    current = node
    for c in username
        if !haskey(current.children, c)
            current.children[c] = TrieNode{V}()
        end
        current = current.children[c]
    end
    current.is_end = true
    current.metadata = meta
end

function autocomplete(
    ac::AutoComplete{V},
    prefix::String,
    N::Int
) where {V}
    current = ac.root
    for c in prefix
        if haskey(current.children, c)
            current = current.children[c]
        else
            return String[]
        end
    end
    results = Vector{Tuple{String,Dict{String,V}}}()
    queue = [(current, prefix)]
    while !isempty(queue) && length(results) < N
        level_results = Tuple{String,Dict{String,V}}[]
        next_queue = []
        for (node, path) in queue
            if node.is_end
                push!(level_results, (path, node.metadata))
            end
            for (c, child) in node.children
                push!(next_queue, (child, path * string(c)))
            end
        end
        sort!(level_results, by = x -> x[2]["sortkey"], rev = true)
        for tup in level_results
            push!(results, tup)
            if length(results) == N
                break
            end
        end
        queue = next_queue
    end
    results
end

function text_encode(data)
    "\\x" * bytes2hex(
        CodecZstd.transcode(CodecZstd.ZstdCompressor, Vector{UInt8}(MsgPack.pack(data))),
    )
end

function save_user_autcompletes()
    logtag("SAVE_AUTOCOMPLETE", "saving user autocompletes")
    source_map = Dict("mal" => 0, "anilist" => 1, "kitsu" => 2, "animeplanet" => 3)
    inv_source_map = Dict(v => k for (k, v) in source_map)
    df = CSV.read("$datadir/profiles.csv", DataFrames.DataFrame)
    d = Dict()
    for v in values(source_map)
        d[v] = Dict{String,Dict{String,Any}}()
    end
    @showprogress for i = 1:DataFrames.nrow(df)
        if ismissing(df.username[i])
            continue
        end
        d[df.source[i]][lowercase(df.username[i])] = Dict(
            "username" => df.username[i],
            "avatar" => df.avatar[i],
            "accessed_at" => df.accessed_at[i],
            "last_online" => df.last_online[i],
            "gender" => df.gender[i],
            "birthday" => df.birthday[i],
            "created_at" => df.created_at[i],
            "sortkey" => (!isnothing(df.avatar[i]) && !ismissing(df.avatar[i]), df.accessed_at[i]),
        )
    end
    acs = Dict(v => AutoComplete(d[v]) for v in values(source_map))
    seen = Dict(v => Set() for v in values(source_map))
    batches = collect(Iterators.partition(1:DataFrames.nrow(df), 1_000_000))
    @showprogress for b = 1:length(batches)
        batch = batches[b]
        records = []
        for i in batch
            if ismissing(df.username[i])
                continue
            end
            source = df.source[i]
            s = inv_source_map[source]
            prefix = ""
            for c in lowercase(df.username[i])
                prefix *= c
                if prefix in seen[source]
                    continue
                end
                push!(seen[source], prefix)
                vals = autocomplete(acs[source], prefix, 10)
                vals = [Dict(k => x[2][k] for k in ["username", "avatar", "last_online", "gender", "birthday", "created_at"]) for x in vals]
                push!(records, (s, prefix, text_encode(vals)))
            end
        end
        ac_df = DataFrames.DataFrame(records, [:source, :prefix, :data])
        CSV.write("$datadir/user_autocomplete.$b.csv", ac_df)
    end
end

function upload_autocompletes()
    logtag("SAVE_AUTOCOMPLETE", "uploading autocompletes")
    qrun(`./save_autocomplete.sh`)
end

import_data()
save_profiles()
save_user_autcompletes()
upload_autocompletes()
