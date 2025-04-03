import CSV
import CodecZstd
import DataFrames
import MsgPack
import ProgressMeter: @showprogress

const datadir = "../../data/finetune"

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
    N::Int;
    sortcol::Union{Nothing,String} = nothing,
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
        if sortcol !== nothing
            sort!(level_results, by = x -> x[2][sortcol], rev = true)
        end
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
                vals = autocomplete(acs[source], prefix, 10; sortcol = "accessed_at")
                vals = [Dict(k => x[2][k] for k in ["username", "avatar"]) for x in vals]
                push!(records, (s, prefix, text_encode(vals)))
            end
        end
        ac_df = DataFrames.DataFrame(records, [:source, :prefix, :data])
        CSV.write("$datadir/user_autocomplete.$b.csv", ac_df)
    end
    cmd = "chmod +x ./save_autocomplete.sh && ./save_autocomplete.sh"
    run(`sh -c $cmd`)
end

save_user_autcompletes()
