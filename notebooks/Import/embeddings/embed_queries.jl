import CSV
import DataFrames
import HTTP
import JLD2
import JSON3
import Memoize: @memoize
import ProgressMeter: @showprogress
include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings/documents"
const secretdir = "../../../secrets"

function get_gcp_access_token()
    run(`gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json`)
    strip(read(`gcloud auth print-access-token`, String))
end
gcp_access_token::String = get_gcp_access_token()

function get_text_embedding(text::AbstractString, gcp_access_token::AbstractString)
    if isempty(text)
        return zeros(Float32, 3072), true
    end
    gcp_project = read("$secretdir/gcp.project.txt", String)
    gcp_region = read("$secretdir/gcp.region.txt", String)
    model_id = "gemini-embedding-001"
    url = "https://$(gcp_region)-aiplatform.googleapis.com/v1/projects/$(gcp_project)/locations/$(gcp_region)/publishers/google/models/$(model_id):predict"
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    payload =
        Dict("instances" => [Dict("task_type" => "RETRIEVAL_QUERY", "content" => text)])
    body = JSON3.write(payload)
    ret = HTTP.post(url, headers, body, status_exception = false)
    if HTTP.iserror(ret)
        logerror("text embedding failed for $(ret.status)")
        return ret, false
    end
    data = JSON3.read(ret.body)
    embs = only(data["predictions"])["embeddings"]
    Float32.(embs["values"]), true
end

function get_text_embedding(text::AbstractString)
    global gcp_access_token
    for attempt = 1:3
        try
            resp, ok = get_text_embedding(text, gcp_access_token)
            @assert ok
            return resp, ok
        catch
            sleep(1)
            gcp_access_token = get_gcp_access_token()
        end
    end
    logerror("get_text_embedding failed for $text")
    zeros(Float32, 3072), false
end

function extract_keywords(document_text::String)::Vector{String}
    pattern = r"(?s)# Keywords\s*\n(.*?)(?=\n\s*#|\z)"
    m = match(pattern, document_text)
    if m === nothing
        return String[]
    end
    keyword_block = m.captures[1]
    keywords = [strip(kw) for kw in split(keyword_block, ',') if !isempty(strip(kw))]
    keywords
end

function get_date_tags(data)
    tags = []
    if !isnothing(data[:season])
        push!(tags, replace(data[:season], "-" => " "))
    end
    if !isnothing(data[:startdate]) && !isempty(data[:startdate])
        if length(data[:startdate]) >= 4
            push!(tags, data[:startdate][1:4])
        end
        start_year = parse(Int, data[:startdate][1:4])
        decade_start = floor(Int, start_year / 10) * 10
        if decade_start >= 2000
            push!(tags, "$(decade_start)s")
        else
            push!(tags, "$(decade_start % 100)s")
        end
    end
    tags
end

function get_queries(json)
    tags = []
    if !isnothing(json[:llm_summary])
        append!(tags, extract_keywords(json[:llm_summary][:text]))
    end
    append!(tags, copy(json[:genres]))
    append!(tags, copy(json[:tags]))
    ret = [(x, "tag") for x in tags]
    for k in [:title, :english_title]
        if !isnothing(json[k]) && !isempty(json[k])
            push!(ret, (json[k], "title"))
        end
    end
    for x in get_date_tags(json[:metadata][:dates])
        try
            push!(ret, (x, "date"))
        catch
            println((json[:metadata][:dates]))
        end
    end
    collect(Set([(lowercase(x), y) for (x, y) in ret]))
end

function get_queries()
    json = open("$datadir/embeddings.json") do f
        JSON3.read(f)
    end
    queries = Set()
    for x in json
        for (q, _) in get_queries(x)
            push!(queries, q)
        end
    end
    queries
end

function embed_queries()
    queries = get_queries()
    run(
        `rclone --retries=10 copyto -Pv r2:rsys/database/import/search_embeddings.jld2 $datadir/search_embeddings.jld2`,
    )
    if ispath("$datadir/search_embeddings.jld2")
        existing_queries = JLD2.load("$datadir/search_embeddings.jld2")["queries"]
    else
        existing_queries = Dict()
    end
    cache_hits = 0
    cache_misses = 0
    d = Dict()
    @showprogress for q in queries
        if q in keys(existing_queries)
            d[q] = existing_queries[q]
            cache_hits += 1
        else
            emb, ok = get_text_embedding(q)
            if ok
                d[q] = emb
            end
            cache_misses += 1
        end
    end
    cache_hitrate = cache_hits / (cache_hits + cache_misses)
    logtag(
        "EMBED_QUERIES",
        "cache hitrate: $cache_hitrate, hits: $cache_hits, misses: $cache_misses",
    )
    JLD2.save("$datadir/search_embeddings.jld2", Dict("queries" => d))
    run(
        `rclone copyto -Pv $datadir/search_embeddings.jld2 r2:rsys/database/import/search_embeddings.jld2`,
    )
end

embed_queries()