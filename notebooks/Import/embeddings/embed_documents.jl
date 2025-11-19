import HTTP
import JSON3
import ProgressMeter: @showprogress

include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings/documents"

function get_embeddings_cache()
    if ispath("$datadir/embeddings.json")
        embeddings = JSON3.read("$datadir/embeddings.json")
    else
        embeddings = []
    end
    cache = Dict()
    for x in embeddings
        v = copy(x[:embedding])
        k = copy(x)
        delete!(k, :embedding)
        cache[k] = v
    end
    cache
end

function json_to_document(x)
    if !isnothing(x[:llm_summary]) && !isempty(strip(x[:llm_summary][:text]))
        return x[:llm_summary][:text]
    end
    doc_parts = String[]
    title = x[:title]
    if !isnothing(title) && !isempty(String(title))
        push!(doc_parts, "# $title")
    end
    length_parts = []
    if !isnothing(x[:metadata][:length][:episodes])
        push!(length_parts, "$(x[:metadata][:length][:episodes]) episodes")
    end
    if !isnothing(x[:metadata][:length][:chapters])
        push!(length_parts, "$(x[:metadata][:length][:chapters]) chapters")
    end
    if !isnothing(x[:metadata][:length][:volumes])
        push!(length_parts, "$(x[:metadata][:length][:volumes]) volumes")
    end
    length_str = join(length_parts, ", ")
    studio_str = join(x[:metadata][:studios], " ")
    genre_str = join(x[:genres], " ")
    alttitle_str = something(x[:english_title], "")
    metadata_str = """
    # Metadata
    - **Type:** $(x[:metadata][:mediatype])
    - **Status:** $(x[:metadata][:status])
    - **Source Material:** $(x[:metadata][:source_material])
    - **Length:** $length_str
    - **Released:** $(x[:metadata][:dates][:startdate]) to $(x[:metadata][:dates][:enddate])
    - **Season:** $(x[:metadata][:dates][:season])
    - **Studios:** $studio_str
    - **Genres:** $genre_str
    - **Alternative Titles:** $alttitle_str"""
    push!(doc_parts, metadata_str)
    premise_str = something(x[:synopsis]..., "")
    push!(doc_parts, "# Premise\n$(premise_str)")
    x[:characters]
    characters = []
    for y in first(x[:characters], 8)
        name = y[:name]
        text = something(y[:description], "")
        push!(characters, "- **$name** $text")
    end
    character_str = join(characters, "\n")
    push!(doc_parts, "# Characters\n$character_str")
    review_str = join([y[:text] for y in first(x[:reviews], 4)], "\n")
    push!(doc_parts, "# Reviews\n$review_str")
    keyword_str = join(x[:tags], ", ")
    push!(doc_parts, "# Keywords\n$keyword_str")
    join(doc_parts, "\n\n")
end

function document_to_sections(x::String)
    valid_sections = Set([
        "Tagline",
        "Metadata",
        "Premise",
        "Analysis",
        "Characters",
        "Reviews",
        "Keywords",
        "Synopsis",
    ])
    data_dict = Dict{String,String}()
    content_buffer = IOBuffer()
    current_key = ""
    lines = split(x, '\n')
    if !isempty(lines)
        title_line = lines[1]
        title = strip(replace(title_line, r"^#\s+" => ""))
        data_dict["Title"] = title
    end
    for line in lines[2:end]
        match_header = match(r"^#\s+([^#].*)", line)
        if !isnothing(match_header)
            if !isempty(current_key)
                data_dict[current_key] = strip(String(take!(content_buffer)))
            end
            new_key = strip(match_header.captures[1])
            if new_key in valid_sections
                current_key = new_key
            else
                current_key = ""
            end
        elseif !isempty(current_key)
            println(content_buffer, line)
        end
    end
    if !isempty(current_key)
        data_dict[current_key] = strip(String(take!(content_buffer)))
    end
    data_dict
end

function sections_to_document(data_dict::Dict{String,String}, section_order::Vector{String})
    md_buffer = IOBuffer()
    for section_name in section_order
        if haskey(data_dict, section_name)
            content = data_dict[section_name]
            if section_name == "Title"
                println(md_buffer, "# ", content)
            else
                println(md_buffer, "# ", section_name)
                println(md_buffer, content)
            end
            println(md_buffer)
        end
    end
    String(take!(md_buffer))
end

function document_to_chunks(text::String)
    # TODO unchunk once embeddings models get >2048 context limit
    sections = document_to_sections(text)
    metadata = sections_to_document(
        sections,
        ["Title", "Metadata", "Keywords", "Synopsis", "Characters", "Premise"],
    )
    analysis = sections_to_document(
        sections,
        [
            "Title",
            "Tagline",
            "Metadata",
            "Premise",
            "Analysis",
            "Characters",
            "Reviews",
            "Keywords",
        ],
    )
    Dict("metadata" => metadata, "analysis" => analysis)
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
    embs = only(data["predictions"])["embeddings"]
    if embs["statistics"]["truncated"]
        num_tokens = embs["statistics"]["token_count"]
        println("truncated embedding with $num_tokens tokens")
    end
    Float32.(embs["values"]), true
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

function save_embeddings()
    emb_cache = get_embeddings_cache()
    jsons = copy(JSON3.read("$datadir/summaries.json"))
    cache_hits = 0
    cache_misses = 0
    @showprogress for json in jsons
        if json in keys(emb_cache)
            json[:embedding] = emb_cache[json]
            cache_hits += 1
        else
            text = json_to_document(json)
            chunks = document_to_chunks(text)
            d_emb = Dict()
            for (k, v) in chunks
                emb = get_embedding(v)
                if isnothing(emb)
                    logerror("text embedding failed for $k")
                    emb = zeros(Float32, 3072)
                end
                d_emb[k] = emb
            end
            json[:embedding] = d_emb
            cache_misses += 1
        end
    end
    cache_hitrate = cache_hits / (cache_hits + cache_misses)
    logtag(
        "EMBED_DOCUMENTS",
        "cache hitrate: $cache_hitrate, hits: $cache_hits, misses: $cache_misses",
    )
    open("$datadir/embeddings.json", "w") do f
        JSON3.write(f, jsons)
    end
    cmd = "rclone --retries=10 copyto $datadir/embeddings.json r2:rsys/database/import/embeddings.json"
    run(`sh -c $cmd`)

end

save_embeddings()