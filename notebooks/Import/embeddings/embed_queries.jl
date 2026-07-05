import CSV
import DataFrames
import HTTP
import JLD2
import JSON3
import Memoize: @memoize
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

function get_text_embedding(text::AbstractString, gcp_access_token::AbstractString)
    if isempty(text)
        return zeros(Float32, 3072), true
    end
    gcp_region = "global"
    url = "https://aiplatform.googleapis.com/v1/projects/$(gcp_project)/locations/$gcp_region/publishers/google/models/$(model_id):embedContent"
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    payload = Dict(
        "content" => Dict(
            "parts" => [Dict("text" => "task: search result | query: $text")],
        ),
    )
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

function get_text_embedding(text::AbstractString)
    for attempt = 1:3
        token = gcp_access_token
        try
            resp, ok = get_text_embedding(text, token)
            @assert ok
            return resp, ok
        catch
            sleep(10)
            update_gcp_access_token(token)
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
    json = open("$datadir/summaries.json") do f
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
    queries = collect(get_queries())
    existing_queries = Dict()
    if ispath("$datadir/query_embeddings.jld2")
        query_embeddings = JLD2.load("$datadir/query_embeddings.jld2")
        cached_model_id = get(query_embeddings, "modelname", nothing) # todo remove guard
        if cached_model_id == model_id
            existing_queries = query_embeddings["queries"]
        else
            logtag(
                "EMBED_QUERIES",
                "using new model $model_id to replace $cached_model_id",
            )
        end
    end
    embeddings = Any[nothing for _ = 1:length(queries)]
    cache_hits = [false for _ = 1:length(queries)]
    cache_misses = [false for _ = 1:length(queries)]
    @showprogress Threads.@threads for i = 1:length(queries)
        q = queries[i]
        if q in keys(existing_queries)
            embeddings[i] = existing_queries[q]
            cache_hits[i] = true
        else
            emb, ok = get_text_embedding(q)
            if ok
                embeddings[i] = emb
                cache_misses[i] = true
            end
        end
    end
    d = Dict(k => v for (k, v) in zip(queries, embeddings) if !isnothing(v))
    num_cache_hits = sum(cache_hits)
    num_cache_misses = sum(cache_misses)
    cache_hitrate = num_cache_hits / (num_cache_hits + num_cache_misses)
    num_fails = length(cache_hits) - num_cache_hits - num_cache_misses
    logtag(
        "EMBED_QUERIES",
        "cache hitrate: $cache_hitrate, hits: $num_cache_hits, misses: $num_cache_misses, fails: $num_fails",
    )
    JLD2.save("$datadir/query_embeddings.jld2", Dict("queries" => d, "modelname" => model_id))
    run(
        `rclone copyto -Pv $datadir/query_embeddings.jld2 r2:rsys/database/import/query_embeddings.jld2`,
    )
end

embed_queries()