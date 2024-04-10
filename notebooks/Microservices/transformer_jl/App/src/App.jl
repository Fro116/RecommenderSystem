module App

import HTTP
import JSON
import Oxygen
import MLUtils
import MsgPack
import NBInclude: @nbinclude
import NNlib: sigmoid
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")
@nbinclude("notebooks/TrainingAlphas/Transformer/Data.ipynb")

function msgpack(d::Dict)::HTTP.Response
    body = MsgPack.pack(d)
    response = HTTP.Response(200, [], body = body)
    HTTP.setheader(response, "Content-Type" => "application/msgpack")
    HTTP.setheader(response, "Content-Length" => string(sizeof(body)))
    response
end

function get_sentences(df::RatingsDataset, cls_tokens, max_seq_length)
    function itemids(uid, medium)
        tokens = [cls_tokens[i] for i = 1:length(ALL_MEDIUMS)]
        tokens[medium+1] = uid
        tokens
    end
    sentence = Vector{wordtype}()
    userid = 0
    @assert all(df.userid .== userid)
    push!(sentence, replace(cls_tokens, :userid, userid))
    # need to sort by updated_at because we're combining multiple_media
    order = sortperm(collect(zip(df.updated_at, -df.update_order)))
    for idx = 1:length(order)
        i = order[idx]
        word = convert(
            wordtype,
            (
                itemids(df.itemid[i], df.medium[i])...,
                df.rating[i],
                df.updated_at[i],
                df.status[i],
                df.source[i],
                df.created_at[i],
                df.started_at[i],
                df.finished_at[i],
                df.progress[i],
                Float32(1 - 1 / (df.repeat_count[i] + 1)),
                Float32(1 - 1 / (df.priority[i] + 1)),
                df.sentiment[i],
                df.sentiment_score[i],
                (length(sentence) - 1) % max_seq_length,
                df.userid[i],
            ),
        )
        push!(sentence, word)
    end
    sentence
end

function get_sentences(
    cls_tokens,
    max_seq_length,
    exclude_ptw::Bool,
    payload::Dict,
)
    function get_df(medium)
        fields = [
            :itemid,
            :rating,
            :updated_at,
            :status,
            :source,
            :created_at,
            :started_at,
            :finished_at,
            :progress,
            :repeat_count,
            :priority,
            :sentiment,
            :sentiment_score,
            :userid,
            :medium,
            :update_order,
        ]
        df = get_raw_split(payload, medium, fields, nothing)
        if exclude_ptw
            df = filter(df, df.status .!= get_status(:plan_to_watch))
        end
        df
    end
    dfs = [get_df(medium) for medium in ALL_MEDIUMS]
    df = reduce(cat, dfs)
    get_sentences(df, cls_tokens, max_seq_length)
end

function tokenize(sentence, config, max_sentence_len)
    tokenize(;
        sentence = copy(sentence),
        userid = 0,
        max_seq_len = config[:max_sequence_length],
        vocab_sizes = config[:vocab_sizes],
        pad_tokens = config[:pad_tokens],
        cls_tokens = config[:cls_tokens],
        mask_tokens = config[:mask_tokens],
        max_sentence_len = max_sentence_len,
    )
end

function tokenize(;
    sentence::Vector{wordtype},
    userid,
    max_seq_len,
    vocab_sizes,
    pad_tokens,
    cls_tokens,
    mask_tokens,
    max_sentence_len
)
    sentence =
        subset_sentence(sentence, min(length(sentence), max_seq_len - 1); recent = true)
    masked_word = mask_tokens
    masked_word = replace(masked_word, :updated_at, 1)
    masked_word = replace(masked_word, :position, length(sentence) - 1)
    masked_word = replace(masked_word, :userid, userid)
    push!(sentence, masked_word)
    pad_len = min(max_sentence_len+1, max_seq_len)
    tokens = get_token_ids(sentence, pad_len, pad_tokens, false)
    positions = [length(sentence) - 1]
    tokens, positions
end

function process_tokens(sentences, config)
    max_sentence_len = maximum(length.(sentences))
    tokens = [tokenize(x, config, max_sentence_len) for x in sentences]
    d = Dict{String,AbstractArray}()
    collate = MLUtils.batch
    for (i, name) in Iterators.enumerate(config.vocab_names)
        d[name] = collate([x[1][i] for x in tokens])
    end
    d["positions"] = collate([x[2] for x in tokens])
    d
end

function get_config()
    version = "v1"
    sourcedir = get_data_path(joinpath("alphas", "all", "Transformer", version, "0"))
    open(joinpath(sourcedir, "config.json")) do f
        d = JSON.parse(f)
        return NamedTuple(Symbol.(keys(d)) .=> values(d))
    end   
end

const CONFIG = get_config()

function process_medialists(payload::Dict)
    sentences = [
        get_sentences(
            CONFIG[:cls_tokens],
            CONFIG[:max_sequence_length],
            include_ptw,
            payload,
        ) for include_ptw in [false, true]
    ]
    process_tokens(sentences, CONFIG)
end

function compute_transformer(payload::Dict, embeddings::Dict, medium::String)
    ret = Dict()
    version = "v1"
    seen = get_raw_split(payload, medium, [:itemid], nothing).itemid
    ptw = get_split(payload, "plantowatch", medium, [:itemid], nothing).itemid
    watched = [x for x in seen if x ∉ Set(ptw)]
    for metric in ALL_METRICS
        M = embeddings["$(medium)_$(metric)"]
        r = M[1] # regular items
        p = M[2] # plantowatch items
        if metric in ["watch", "plantowatch"]
            r = exp.(r)
            r[seen.+1] .= 0
            r = r ./ sum(r)
            p = exp.(p)
            p[watched.+1] .= 0
            p = p ./ sum(p)
        elseif metric == "drop"
            r = sigmoid.(r)
            p = sigmoid.(p)
        elseif metric == "rating"
            nothing
        else
            @assert false
        end
        e = copy(r)
        e[ptw.+1] .= p[ptw.+1]
        ret["$medium/Transformer/$version/$metric"] = e[1:num_items(medium)]
    end
    ret
end

function wake(req::HTTP.Request)
    msgpack(Dict("success" => true))
end

function process(req::HTTP.Request)
    payload = MsgPack.unpack(req.body)
    msgpack(process_medialists(payload))
end

function compute(req::HTTP.Request)
    d = MsgPack.unpack(req.body)
    params = Oxygen.queryparams(req)
    payload = d["payload"]
    embeddings = d["embedding"]
    alpha = compute_transformer(payload, embeddings, params["medium"])
    msgpack(alpha)
end

function precompile_run(running::Bool, port::Int, query::String)
    if running
        return HTTP.get("http://localhost:$port$query")
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        r = HTTP.Request("GET", query, [], "")
        return fn(r)
    end
end

function precompile_run(running::Bool, port::Int, query::String, data::Vector{UInt8})
    if running
        return HTTP.post(
            "http://localhost:$port$query",
            [("Content-Type", "application/msgpack")],
            data,
        )
    else
        name = split(query[2:end], "?")[1]
        fn = getfield(App, Symbol(name))
        req = HTTP.Request("POST", query, [("Content-Type", "application/msgpack")], data)
        return fn(req)
    end
end

function precompile(running::Bool, port::Int)
    while true
        try
            r = precompile_run(running, port, "/wake")
            if MsgPack.unpack(r.body)["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end
    
    payload = MsgPack.pack(
        Dict(
            "anime" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[1],
                "priority" => Int32[0],
                "status" => Int32[6],
                "progress" => Float32[1.0],
                "updated_at" => Float32[1.0],
                "started_at" => Float32[0.0],
                "repeat_count" => Int32[0],
                "owned" => Int32[0],
                "sentiment" => Int32[0],
                "itemid" => Int32[0],
                "finished_at" => Float32[0.0],
                "source" => Int32[0],
                "userid" => Int32[0],
            ),
            "manga" => Dict(
                "created_at" => Float32[0.0],
                "rating" => Float32[1.0],
                "update_order" => Int32[0],
                "sentiment_score" => Float32[0.0],
                "medium" => Int32[0],
                "priority" => Int32[0],
                "status" => Int32[6],
                "progress" => Float32[1.0],
                "updated_at" => Float32[1.0],
                "started_at" => Float32[0.0],
                "repeat_count" => Int32[0],
                "owned" => Int32[0],
                "sentiment" => Int32[0],
                "itemid" => Int32[0],
                "finished_at" => Float32[0.0],
                "source" => Int32[0],
                "userid" => Int32[0],
            ),
        ),
    )
    precompile_run(running, port, "/process", payload)

    for medium in ALL_MEDIUMS  
        d = Dict()
        d["payload"] = MsgPack.unpack(payload)        
        d["embedding"] = Dict(
            "$(medium)_$(metric)" => [ones(Float32, num_items(medium)) for _ = 1:2]
            for metric in ALL_METRICS
        )
        precompile_run(running, port, "/compute?medium=$medium", MsgPack.pack(d))
    end
end

end