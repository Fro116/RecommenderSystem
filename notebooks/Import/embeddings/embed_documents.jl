import HTTP
import JLD2
import JSON3
import ProgressMeter: @showprogress, next!
include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings"
const secretdir = "../../../secrets"
const gcp_project = read("$secretdir/gcp.project.txt", String)
const model_id = "gemini-embedding-2"
const gcp_lock = ReentrantLock()

gcp_access_token::String = ""
function update_gcp_access_token(token::String)
    global gcp_access_token
    lock(gcp_lock) do
        if token != gcp_access_token
            return
        end
        run(`gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json`)
        gcp_access_token = strip(read(`gcloud auth print-access-token`, String))
    end
end
update_gcp_access_token(gcp_access_token)

function get_embeddings_cache()
    if !ispath("$datadir/document_embeddings.jld2")
        return Dict()
    end
    embeddings = JLD2.load("$datadir/document_embeddings.jld2")
    cached_model_id = embeddings["modelname"]
    if cached_model_id != model_id
        logtag("EMBED_DOCUMENTS", "using new model $model_id to replace $cached_model_id")
        return Dict()
    end
    Dict(v["text"] => v["embedding"] for (_, v) in embeddings["documents"] if v["success"])
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
        "Queries",
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

function get_embedding(
    title::AbstractString,
    text::AbstractString,
    gcp_access_token::AbstractString,
)
    @assert !(isempty(title) && isempty(text))
    gcp_region = "global"
    url = "https://aiplatform.googleapis.com/v1/projects/$(gcp_project)/locations/$gcp_region/publishers/google/models/$(model_id):embedContent"
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    payload =
        Dict("content" => Dict("parts" => [Dict("text" => "title: $title | text: $text")]))
    body = JSON3.write(payload)
    ret = HTTP.post(url, headers, body, status_exception = false)
    if HTTP.iserror(ret)
        logerror("text embedding failed for $(ret.status)")
        return ret, false
    end
    data = JSON3.read(ret.body)
    if get(data, "truncated", false)
        num_tokens = data["usageMetadata"]["totalTokenCount"]
        logerror("truncated embedding with $num_tokens tokens")
    end
    Float32.(data["embedding"]["values"]), true
end

function get_embedding(title::AbstractString, text::AbstractString)
    for retry = 1:3
        token = gcp_access_token
        ret, ok = get_embedding(title, text, token)
        if ok
            return ret, true
        else
            sleep(10)
            update_gcp_access_token(token)
        end
    end
    logerror("text embedding failed for $title")
    zeros(Float32, 3072), false
end

function save_embeddings()
    existing_embeddings = get_embeddings_cache()
    jsons = copy(JSON3.read("$datadir/summaries.json"))
    texts = ["" for _ = 1:length(jsons)]
    embeddings = Any[nothing for _ = 1:length(jsons)]
    cache_hits = [false for _ = 1:length(jsons)]
    cache_misses = [false for _ = 1:length(jsons)]
    @showprogress Threads.@threads for i = 1:length(jsons)
        json = jsons[i]
        text = sections_to_document(
            document_to_sections(json_to_document(json)),
            [
                "Title",
                "Tagline",
                "Metadata",
                "Premise",
                "Analysis",
                "Synopsis",
                "Characters",
                "Keywords",
                "Reviews",
            ],
        )
        texts[i] = text
        if text in keys(existing_embeddings)
            embeddings[i] = Dict(
                "embedding" => existing_embeddings[text],
                "keys" => json[:keys],
                "success" => true,
            )
            cache_hits[i] = true
        else
            title = json[:title]
            emb, ok = get_embedding(title, text)
            embeddings[i] = Dict("embedding" => emb, "keys" => json[:keys], "success" => ok)
            if ok
                cache_misses[i] = true
            end
        end
    end
    d = Dict()
    for (t, e) in zip(texts, embeddings)
        d[e["keys"]] =
            Dict("text" => t, "embedding" => e["embedding"], "success" => e["success"])
    end
    num_cache_hits = sum(cache_hits)
    num_cache_misses = sum(cache_misses)
    cache_hitrate = num_cache_hits / (num_cache_hits + num_cache_misses)
    num_fails = length(cache_hits) - num_cache_hits - num_cache_misses
    logtag(
        "EMBED_DOCUMENTS",
        "cache hitrate: $cache_hitrate, hits: $num_cache_hits, misses: $num_cache_misses, fails: $num_fails",
    )
    JLD2.save(
        "$datadir/document_embeddings.jld2",
        Dict("documents" => d, "modelname" => model_id),
    )
    run(
        `rclone copyto -Pv $datadir/document_embeddings.jld2 r2:rsys/database/import/document_embeddings.jld2`,
    )
end

save_embeddings()
