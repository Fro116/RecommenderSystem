module App

import HTTP
import JSON
import Oxygen
import MLUtils
import NBInclude: @nbinclude
import NNlib: sigmoid
@nbinclude("notebooks/TrainingAlphas/AlphaBase.ipynb")
@nbinclude("notebooks/TrainingAlphas/Transformer/Data.ipynb")

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

function tokenize(sentence, config)
    tokenize(;
        sentence = copy(sentence),
        userid = 0,
        max_seq_len = config[:max_sequence_length],
        vocab_sizes = config[:vocab_sizes],
        pad_tokens = config[:pad_tokens],
        cls_tokens = config[:cls_tokens],
        mask_tokens = config[:mask_tokens],
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
)
    sentence =
        subset_sentence(sentence, min(length(sentence), max_seq_len - 1); recent = true)
    masked_word = mask_tokens
    masked_word = replace(masked_word, :updated_at, 1)
    masked_word = replace(masked_word, :position, length(sentence) - 1)
    masked_word = replace(masked_word, :userid, userid)
    push!(sentence, masked_word)
    tokens = get_token_ids(sentence, max_seq_len, pad_tokens, false)
    positions = [length(sentence) - 1]
    tokens, positions
end

function process_tokens(sentences, config)
    tokens = [tokenize(x, config) for x in sentences]
    d = Dict{String,AbstractArray}()
    collate = MLUtils.batch
    for (i, name) in Iterators.enumerate(config.vocab_names)
        d[name] = collate([x[1][i] for x in tokens])
    end
    d["positions"] = collate([x[2] for x in tokens])
    d
end

function process_medialists(payload::Dict)
    version = "v1"
    sourcedir = get_data_path(joinpath("alphas", "all", "Transformer", version, "0"))
    f = open(joinpath(sourcedir, "config.json"))
    d = JSON.parse(f)
    config = NamedTuple(Symbol.(keys(d)) .=> values(d))
    close(f)

    sentences = [
        get_sentences(
            config[:cls_tokens],
            config[:max_sequence_length],
            include_ptw,
            payload,
        ) for include_ptw in [false, true]
    ]
    process_tokens(sentences, config)
end

function compute_alpha(payload::Dict, embeddings::Dict)
    ret = Dict()
    version = "v1"
    for medium in ALL_MEDIUMS
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
            end
            e = copy(r)
            e[ptw.+1] .= p[ptw.+1]
            ret["$medium/Transformer/$version/$metric"] = e
        end
    end
    ret
end

function wake(req::HTTP.Request)
    Oxygen.json(Dict("success" => true))
end

function process(req::HTTP.Request)
    payload = JSON.parse(String(req.body))
    Oxygen.json(process_medialists(payload))
end

function compute(req::HTTP.Request)
    d = JSON.parse(String(req.body))
    payload = d["payload"]
    embeddings = d["embeddings"]
    Oxygen.json(compute_alpha(payload, embeddings))
end

function precompile(port::Int)
    while true
        try
            r = HTTP.get("http://localhost:$port/wake")
            json = JSON.parse(String(copy(r.body)))
            print(json)
            if json["success"] == true
                break
            end
        catch
            @warn "service down"
            sleep(1)
        end
    end
    
    payload = (
        "{\"anime\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0]," *
        "\"update_order\":[0],\"sentiment_score\":[0],\"medium\":[1],\"backward_order\":[1]," *
        "\"priority\":[0],\"progress\":[1.0],\"forward_order\":[1],\"status\":[6]," *
        "\"updated_at\":[1.0],\"started_at\":[0.0],\"repeat_count\":[0],\"owned\":[0]," *
        "\"sentiment\":[0],\"finished_at\":[0.0],\"source\":[0],\"unit\":[1],\"userid\":[0]}," *
        "\"manga\":{\"mediaid\":[0],\"created_at\":[0],\"rating\":[1.0],\"update_order\":[0]," *
        "\"sentiment_score\":[0],\"medium\":[0],\"backward_order\":[1],\"priority\":[0]," *
        "\"progress\":[1.0],\"forward_order\":[1],\"status\":[6],\"updated_at\":[1.0]," *
        "\"started_at\":[0],\"repeat_count\":[0],\"owned\":[0],\"sentiment\":[0]," *
        "\"finished_at\":[0],\"source\":[0],\"unit\":[1],\"userid\":[0]}}"
    )
    HTTP.post(
        "http://localhost:$port/process",
        [("Content-Type", "application/json")],
        payload,
    )

    embeddings = Dict()
    for medium in ALL_MEDIUMS
        for metric in ALL_METRICS
            embeddings["$(medium)_$(metric)"] = ones(Float32, num_items(medium), 2)
        end
    end
    d = Dict()
    d["payload"] = JSON.parse(payload)
    d["embeddings"] = embeddings
    HTTP.post(
        "http://localhost:$port/compute",
        [("Content-Type", "application/json")],
        JSON.json(d),
    )    
end

end