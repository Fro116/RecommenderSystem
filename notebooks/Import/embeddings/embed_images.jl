import CSV
import Base64
import DataFrames
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

function get_image_embedding(fn::AbstractString, gcp_access_token::AbstractString)
    @assert endswith(fn, ".webp")
    image_url = "https://cdn.recs.moe/images/cards/$fn"
    image_resp = HTTP.get(image_url)
    image = Base64.base64encode(image_resp.body)
    gcp_region = "global"
    url = "https://aiplatform.googleapis.com/v1/projects/$(gcp_project)/locations/$gcp_region/publishers/google/models/$(model_id):embedContent"
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    payload = Dict(
        "content" => Dict(
            "parts" => [
                Dict("inline_data" =>
                        Dict("mime_type" => "image/webp", "data" => image)),
            ],
        ),
    )
    body = JSON3.write(payload)
    ret = HTTP.post(url, headers, body, status_exception = false)
    if HTTP.iserror(ret)
        logerror("image embedding failed for $(ret.status)")
        return ret, false
    end
    data = JSON3.read(ret.body)
    Float32.(data["embedding"]["values"]), true
end

function get_image_embedding(fn::AbstractString)
    for attempt = 1:3
        token = gcp_access_token
        try
            resp, ok = get_image_embedding(fn, token)
            @assert ok
            return resp, ok
        catch
            sleep(10)
            update_gcp_access_token(token)
        end
    end
    logerror("get_image_embedding failed for $fn")
    zeros(Float32, 3072), false
end

function embed_images()
    df = CSV.read(
        "$datadir/images.csv",
        DataFrames.DataFrame,
        types = Dict("imageid" => String),
    )
    image_hashes = collect(Dict(df.imagehash .=> df.filename))
    existing_images = Dict()
    if ispath("$datadir/image_embeddings.jld2")
        image_embeddings = JLD2.load("$datadir/image_embeddings.jld2")
        cached_model_id = image_embeddings["modelname"]
        if cached_model_id == model_id
            existing_images = image_embeddings["images"]
        else
            logtag("EMBED_IMAGES", "using new model $model_id to replace $cached_model_id")
        end
    end
    embeddings = Any[nothing for _ = 1:length(image_hashes)]
    cache_hits = [false for _ = 1:length(image_hashes)]
    cache_misses = [false for _ = 1:length(image_hashes)]
    @showprogress Threads.@threads for i = 1:length(image_hashes)
        image_hash, fn = image_hashes[i]
        if image_hash in keys(existing_images)
            embeddings[i] = existing_images[image_hash]
            cache_hits[i] = true
        else
            emb, ok = get_image_embedding(fn)
            if ok
                embeddings[i] = emb
                cache_misses[i] = true
            end
        end
    end
    d = Dict(
        image_hash => emb for
        ((image_hash, _), emb) in zip(image_hashes, embeddings) if !isnothing(emb)
    )
    num_cache_hits = sum(cache_hits)
    num_cache_misses = sum(cache_misses)
    cache_hitrate = num_cache_hits / (num_cache_hits + num_cache_misses)
    num_fails = length(cache_hits) - num_cache_hits - num_cache_misses
    logtag(
        "EMBED_IMAGES",
        "cache hitrate: $cache_hitrate, hits: $num_cache_hits, misses: $num_cache_misses, fails: $num_fails",
    )
    JLD2.save(
        "$datadir/image_embeddings.jld2",
        Dict("images" => d, "modelname" => model_id),
    )
    run(
        `rclone copyto -Pv $datadir/image_embeddings.jld2 r2:rsys/database/import/image_embeddings.jld2`,
    )
end

embed_images()
