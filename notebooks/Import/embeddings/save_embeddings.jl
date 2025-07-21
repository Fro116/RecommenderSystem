import CodecZstd
import CSV
import DataFrames
import HTTP
import JSON3
import MsgPack
import ProgressMeter: @showprogress

include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings/item_similarity"
const mediums = ["manga", "anime"]
const sources = ["mal", "anilist", "kitsu", "animeplanet"]

function download_data()
    logtag("SAVE_EMBEDDINGS", "downloading data")
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = "rclone --retries=10 copyto r2:rsys/database/import"
    files = vcat(
        ["$m.groups.csv" for m in mediums],
        ["$(s)_$(m).csv" for s in sources for m in mediums],
        ["item_text_embeddings.$m.json" for m in [0, 1]],
    )
    for fn in files
        cmd = "$retrieval/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
end

function get_media(source, medium::String)
    fn = "$datadir/$(source)_$(medium).csv"
    df = CSV.read(fn, DataFrames.DataFrame, ntasks = 1)
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
    groups =
        CSV.read(fn, DataFrames.DataFrame, types = Dict("itemid" => String), ntasks = 1)
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

function format_synopsis(x::Union{String,Missing})
    if ismissing(x) || occursin("No synopsis has been added", x)
        return missing
    end
    text = String(x)
    text = replace(text, r"<[^>]*>" => " ")
    lines = split(text, '\n')
    filter!(!isempty, lines)
    if !isempty(lines)
        last_line = strip(lines[end])
        if occursin(r"\[Written by|^\(Source:"i, last_line)
            pop!(lines)
        end
    end
    text = join(lines, " ")
    text = replace(text, r"\s+" => " ") |> strip
    isempty(text) ? missing : text
end

function get_embedding_args(df)
    args = Dict()
    for x in eachrow(df)
        if x.matchedid in keys(args) || x.matchedid == 0
            continue
        end
        d = Dict(
            "source" => x.source,
            "itemid" => x.itemid,
            "title" => x.title,
            "alttitle" => x.english_title,
            "mediatype" => x.mediatype,
            "genres" => x.genres,
            "synopsis" => format_synopsis(x.synopsis),
        )
        args[x.matchedid] = d
    end
    collect(values(args))
end

function metadata_to_document(item_metadata::Dict)
    doc_parts = String[]
    title = get(item_metadata, "title", missing)
    if !ismissing(title) && !isempty(String(title))
        push!(doc_parts, "Title: $(String(title))")
    end
    alt_title = get(item_metadata, "alttitle", missing)
    if !ismissing(alt_title) && !isempty(String(alt_title))
        push!(doc_parts, "Alternate Title: $(String(alt_title))")
    end
    genres_json = get(item_metadata, "genres", missing)
    if !ismissing(genres_json) && !isempty(String(genres_json))
        try
            genres_list = JSON3.read(String(genres_json), Vector{String})
            if !isempty(genres_list)
                formatted_genres = join(genres_list, ", ")
                push!(doc_parts, "Genres: $(formatted_genres)")
            end
        catch e
            println(stderr, "Warning: Could not parse genres JSON for item. Error: $e")
        end
    end
    synopsis = get(item_metadata, "synopsis", missing)
    if !ismissing(synopsis) && !isempty(String(synopsis))
        cleaned_synopsis = replace(String(synopsis), r"\s+" => " ") |> strip
        push!(doc_parts, "Synopsis: $(cleaned_synopsis)")
    end
    media_type = get(item_metadata, "mediatype", missing)
    if !ismissing(media_type) && !isempty(String(media_type))
        push!(doc_parts, "Media Type: $(String(media_type))")
    end
    final_parts = map(part -> rstrip(part, '.'), doc_parts)
    return join(filter(!isempty, final_parts), ". ") * "."
end


function get_media_groups(medium::Int)
    df = get_media_groups(Dict(0 => "manga", 1 => "anime")[medium])
end

const secretdir = "../../../secrets"
run(`gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json`)
const gcp_project = read("$secretdir/gcp.project.txt", String)
const gcp_region = "us-central1"
get_gcp_access_token() = strip(read(`gcloud auth print-access-token`, String))
gcp_access_token::String = get_gcp_access_token()

function get_embedding(text::AbstractString, gcp_access_token::AbstractString)
    if isempty(text)
        return zeros(Float32, 3072), true
    end
    model_id = "gemini-embedding-001"
    url = "https://$(gcp_region)-aiplatform.googleapis.com/v1/projects/$(gcp_project)/locations/$(gcp_region)/publishers/google/models/$(model_id):predict"
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    payload =
        Dict("instances" => [Dict("task_type" => "RETRIEVAL_DOCUMENT", "content" => text)])
    body = JSON3.write(payload)
    ret = HTTP.post(url, headers, body, status_exception = false)
    if HTTP.iserror(ret)
        logerror("text embedding failed for $(ret.status)")
        return ret, false
    end
    data = JSON3.read(ret.body)
    Float32.(data["predictions"][1]["embeddings"]["values"]), true
end

function get_embedding(text::AbstractString)
    global gcp_access_token
    for retry = 1:3
        ret, ok = get_embedding(text, gcp_access_token)
        if ok
            return ret
        else
            gcp_access_token = get_gcp_access_token()
            sleep(1)
        end
    end
    nothing
end

function save_embeddings(medium::Int)
    logtag("SAVE_EMBEDDINGS", "saving embeddings for medium $medium")
    emb_cache = Dict()
    fn = "$datadir/item_text_embeddings.$medium.json"
    if ispath(fn)
        for x in JSON3.read(read(fn, String))
            emb_cache[x["embedding_text"]] = x["embedding"]
        end
    else
        logerror("previous embeddings $fn not found")
    end
    df = get_media_groups(medium)
    embedding_args = get_embedding_args(df)
    embeddings = []
    @showprogress for x in embedding_args
        x["embedding_text"] = metadata_to_document(x)
        if x["embedding_text"] in keys(emb_cache)
            emb = emb_cache[x["embedding_text"]]
        else
            emb = get_embedding(x["embedding_text"])
        end
        if isnothing(emb)
            logerror("text embedding failed for $x")
            continue
        end
        d = merge(x, Dict("embedding" => emb))
        push!(embeddings, d)
    end
    open(fn, "w") do f
        JSON3.write(f, embeddings)
    end
    run(
        `rclone --retries=10 copyto $fn r2:rsys/database/import/item_text_embeddings.$medium.json`,
    )
end

download_data()
for m in [0, 1]
    save_embeddings(m)
end