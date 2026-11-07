const gcp_project = read("$secretdir/gcp.project.txt", String)
const model_id = "gemini-embedding-2"
const gcp_lock = ReentrantLock()
GCP_ACCESS_TOKEN::String = ""
GCP_ACCESS_CREDIT::Int = 1_000_000

function update_gcp_access_token(token::String)
    global GCP_ACCESS_TOKEN
    lock(gcp_lock) do
        if token != GCP_ACCESS_TOKEN
            return
        end
        try
            run(pipeline(`gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json`, stderr=devnull))
            GCP_ACCESS_TOKEN = strip(read(`gcloud auth print-access-token`, String))
        catch e
            logerror("update_gcp_access_token failed: $e")
        end
    end
end
update_gcp_access_token(GCP_ACCESS_TOKEN)

function refresh_gcp_access()
    global GCP_ACCESS_CREDIT
    lock(gcp_lock) do
        GCP_ACCESS_CREDIT = 1_000_000
    end
    update_gcp_access_token(GCP_ACCESS_TOKEN)
end
Threads.@spawn @periodic "RUN_TRAINING" 3200 @handle_errors refresh_gcp_access()

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
        "content" =>
            Dict("parts" => [Dict("text" => "task: search result | query: $text")]),
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
    text = lowercase(text)
    if length(text) > 1000
        logerror("get_text_embedding: truncating to 1000 chars")
        text = first(text, 1000)
    end
    global GCP_ACCESS_CREDIT
    has_credit = lock(gcp_lock) do
        est_tokens = div(length(text), 4, RoundUp)
        if GCP_ACCESS_CREDIT < est_tokens
            return false
        end
        GCP_ACCESS_CREDIT -= est_tokens
        true
    end
    if !has_credit
        return zeros(Float32, 3072), false
    end
    for attempt = 1:3
        token = GCP_ACCESS_TOKEN
        try
            resp, ok = get_text_embedding(text, token)
            @assert ok
            return resp, ok
        catch
            update_gcp_access_token(token)
        end
    end
    logerror("get_text_embedding failed for $text")
    zeros(Float32, 3072), false
end