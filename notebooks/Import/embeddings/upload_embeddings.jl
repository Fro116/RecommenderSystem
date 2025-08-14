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
        v = x[:embedding]
        k = copy(x)
        delete!(k, :embedding)
        cache[k] = v
    end
    cache
end

function json_to_document(x)
    if !isnothing(x[:llm_summary])
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
    - **Alternative Titles:** $alttitle_str
    - *Type: $(x[:metadata][:mediatype])"""
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
    recommendations = []
    for y in first(x[:recommendations], 3)
        title = y[:title]
        reason = join(first(y[:reasons], 4), " ")
        push!(recommendations, "- **$title** $reason")
    end
    recommendation_str = join(recommendations, "\n")
    push!(doc_parts, "# Recommendations\n$recommendation_str")
    keyword_str = join(x[:tags], ", ")
    push!(doc_parts, "# Keywords\n$keyword_str")
    join(doc_parts, "\n\n")
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

function save_embeddings()
    emb_cache = get_embeddings_cache()
    jsons = copy(JSON3.read("$datadir/summaries.json"))
    @showprogress for json in jsons
        if json in keys(emb_cache)
            emb = emb_cache[json]
        else
            @assert false
            text = json_to_document(json)
            emb = get_embedding(text)
            if isnothing(emb)
                logerror("text embedding failed for $x")
                continue
            end
            json[:embedding] = emb
        end
    end
    open("$datadir/embeddings.json", "w") do f
        JSON3.write(f, jsons)
    end
end

save_embeddings()