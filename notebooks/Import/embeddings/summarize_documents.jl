import Glob
import HTTP
import JSON3
import Random
include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings/documents"
const secretdir = "../../../secrets"

function truncate_array(arr, p)
    n = Int(round(length(arr) * p))
    arr[1:n]
end

function truncate_item!(item, p)
    item["reviews"] = truncate_array(item["reviews"], p)
    for x in item["recommendations"]
        x["reasons"] = truncate_array(x["reasons"], p)
    end
end

function truncate_items!(items)
    max_prompt_tokens = 10_000
    max_token_limit = 1e6
    chars_per_token = 4
    capacity = 0.95
    max_character_limit = (max_token_limit - max_prompt_tokens) * chars_per_token * capacity
    for (k, v) in items
        while length(JSON3.write(v)) > max_character_limit
            truncate_item!(v, 0.9)
        end
    end
end

function upload_documents()
    for medium in ["manga", "anime"]
        prompt = read("$secretdir/gcp.prompt.txt", String)
        items = JSON3.read("$datadir/$medium.json", Dict)
        truncate_items!(items)
        mkpath("$datadir/$medium")
        for (k, v) in items
            open("$datadir/$medium/$k.json", "w") do f
                JSON3.write(f, v)
            end
        end
    end
    bucket = read("$secretdir/gcp.bucket.txt", String)
    cmds = [
        "gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json",
        "(gcloud storage rsync --quiet --delete-unmatched-destination-objects $datadir/anime $bucket/embeddings/documents/anime 2>&1 | grep -v Copying)",
        "(gcloud storage rsync --quiet --delete-unmatched-destination-objects $datadir/manga $bucket/embeddings/documents/manga 2>&1 | grep -v Copying)",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function generate_input_json(prompt::String, filename::String)
    Dict(
        "request" => Dict(
            "contents" => [
                Dict(
                    "role" => "user",
                    "parts" => [
                        Dict("text" => prompt),
                        Dict(
                            "fileData" => Dict(
                                "fileUri" => filename,
                                "mimeType" => "text/plain",
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )
end

function get_generations_cache()
    if ispath("$datadir/embeddings.json")
        generations = JSON3.read("$datadir/embeddings.json")
    else
        generations = []
    end
    cached_generations = Dict()
    for x in generations
        v = x[:llm_summary]
        k = copy(x)
        delete!(k, :llm_summary)
        delete!(k, :embedding)
        cached_generations[k] = v
    end
    cached_generations
end

function upload_batch_job()
    bucket = read("$secretdir/gcp.bucket.txt", String)
    prompt = read("$secretdir/gcp.prompt.txt", String)
    generations_cache = get_generations_cache()
    jsons = []
    for medium in ["manga", "anime"]
        for fn in readdir("$datadir/$medium")
            input_json = JSON3.read("$datadir/$medium/$fn")
            if input_json in keys(generations_cache)
                continue
            end
            filename = "$bucket/embeddings/documents/$medium/$fn"
            json = generate_input_json(prompt, filename)
            push!(jsons, json)
        end
    end
    Random.shuffle!(jsons)
    mkpath("$datadir/batch")
    max_requests_per_job = 200_000
    @assert length(jsons) <= max_requests_per_job
    open("$datadir/batch/job.jsonl", "w") do f
        for json in jsons
            JSON3.write(f, json)
            write(f, "\n")
        end
    end
    cmds = [
        "gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json",
        "gcloud storage rm --recursive $bucket/embeddings/documents/batch",
        "gcloud storage rsync --quiet $datadir/batch $bucket/embeddings/documents/batch",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
    length(jsons)
end

function get_gcp_access_token()
    run(`gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json`)
    gcp_project = read("$secretdir/gcp.project.txt", String)
    read("$secretdir/gcp.region.txt", String)
    strip(read(`gcloud auth print-access-token`, String))
end

function queue_batch_job()
    region = read("$secretdir/gcp.region.txt", String)
    project = read("$secretdir/gcp.project.txt", String)
    url = "https://$(region)-aiplatform.googleapis.com/v1/projects/$project/locations/$region/batchPredictionJobs"
    bucket = read("$secretdir/gcp.bucket.txt", String)
    t = Int(round(time()))
    payload = Dict(
        "displayName" => "batch_job_$t",
        # TODO add model routing to use flash for less popular series
        "model" => "publishers/google/models/gemini-2.5-pro",
        "inputConfig" => Dict(
            "instancesFormat" => "jsonl",
            "gcsSource" =>
                Dict("uris" => "$bucket/embeddings/documents/batch/job.jsonl"),
        ),
        "outputConfig" => Dict(
            "predictionsFormat" => "jsonl",
            "gcsDestination" =>
                Dict("outputUriPrefix" => "$bucket/embeddings/documents/batch"),
        ),
    )
    gcp_access_token = get_gcp_access_token()
    headers = Dict(
        "Authorization" => "Bearer $gcp_access_token",
        "Content-Type" => "application/json",
    )
    body = JSON3.write(payload)
    ret = HTTP.post(url, headers, body, status_exception = false)
    resp = JSON3.parse(String(copy(ret.body)))
    batch_job_id = split(resp[:name], "/")[end]
end

function wait_on_batch_job(batch_job_id)
    region = read("$secretdir/gcp.region.txt", String)
    project = read("$secretdir/gcp.project.txt", String)
    is_finished = false
    while !is_finished
        logtag("SUMMARIZE_DOCUMENTS", "waiting on batch job")
        sleep(600)
        gcp_access_token = get_gcp_access_token()
        headers = Dict(
            "Authorization" => "Bearer $gcp_access_token",
            "Content-Type" => "application/json",
        )
        url = "https://$(region)-aiplatform.googleapis.com/v1/projects/$project/locations/$region/batchPredictionJobs/$batch_job_id"
        ret = HTTP.get(url, headers, status_exception = false)
        resp = JSON3.parse(String(copy(ret.body)))
        is_finished = resp["state"] == "JOB_STATE_SUCCEEDED"
    end
    bucket = read("$secretdir/gcp.bucket.txt", String)
    cmds = [
        "gcloud auth login --quiet --cred-file=$secretdir/gcp.auth.json",
        "gcloud storage rsync -r --quiet --exclude \".*incremental_predictions.*\" $bucket/embeddings/documents/batch $datadir/batch",
    ]
    cmd = join(cmds, " && ")
    run(`sh -c $cmd`)
end

function save_generations()
    generations_cache = get_generations_cache()
    fns = Glob.glob("$datadir/batch/*/predictions.jsonl")
    if !isempty(fns)
        jsons = JSON3.parse.(readlines(only(fns)))
        fails = 0
        for json in jsons
            req = only(json[:request][:contents])
            prompt = req[:parts][1][:text]
            input_fn = req[:parts][2][:fileData][:fileUri]
            medium, base = split(input_fn, "/")[end-1:end]
            input_json = copy(JSON3.read("$datadir/$medium/$base"))
            try
                text = only(only(json[:response][:candidates])[:content][:parts])[:text]
                modelname = json[:response][:modelVersion]
                generations_cache
                generations_cache[input_json] =
                    Dict(:text => text, :prompt => prompt, :modelname => modelname)
            catch
                generations_cache[input_json] = nothing
                fails += 1
            end
            push!(generations, input_json)
        end
        if fails > 0
            failure_perc = fails / length(jsons)
            logerror(
                "generation failed for $failure_perc = ($fails / $(length(jsons))) items",
            )
            @assert failure_perc < 0.1
        end
    end
    jsons = []
    for medium in ["manga", "anime"]
        for fn in readdir("$datadir/$medium")
            json = copy(JSON3.read("$datadir/$medium/$fn"))
            json[:llm_summary] = generations_cache[json]
            push!(jsons, json)
        end
    end
    open("$datadir/summaries.json", "w") do f
        JSON3.write(f, jsons)
    end
end

function summarize_documents()
    upload_documents()
    num_jobs = upload_batch_job()
    logtag("[SUMMARIZE_DOCUMENTS]", "running batch job on $num_jobs documents")
    @assert num_jobs == 0
    if num_jobs > 0
        batch_job_id = queue_batch_job()
        wait_on_batch_job(batch_job_id)
    end
    save_generations()
end

summarize_documents()