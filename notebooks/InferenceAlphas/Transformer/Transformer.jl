#   Transformer
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#     •  See the corresponding file in ../../TrainingAlphas for more
#        details

import NBInclude: @nbinclude
if !@isdefined TRANSFORMER_IFNDEF
    TRANSFORMER_IFNDEF=true
    source_name = "Transformer";
    import Random
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Transformer/Reference/Include.ipynb");

    function get_training_data(include_ptw, cls_tokens)
        function get_df(content)
            df = get_raw_recommendee_split(content)
            if content != "explicit"
                df.rating .= 11
            end
            df
        end

        contents = ["explicit", "implicit"]
        if include_ptw
            push!(contents, "ptw")
        end
        sentences = Dict{Int32,Vector{wordtype}}()
        df = reduce(cat, [get_df(content) for content in contents])
        order = sortperm(df.timestamp)
        for i in order
            if df.user[i] ∉ keys(sentences)
                sentences[df.user[i]] = [replace(cls_tokens, :user, df.user[i])]
            end
            # handle timestamps from the future # TODO
            ts = df.timestamp[i]
            if ts > 1f0
                ts = 1
            end
            word = encode_word(
                df.item[i],
                df.rating[i],
                ts,
                df.status[i],
                df.completion[i],
                df.user[i],
                length(sentences[df.user[i]]),
            )
            push!(sentences[df.user[i]], word)
        end
        sentences[1]
    end
    
    function crop_sentence(s, task, max_seq_len, mask_tokens)
        rng = Random.GLOBAL_RNG
        if task == "random"
            s = subset_sentence(s, max_seq_len - 1; recent = false, rng = rng)
            masked_word = mask_tokens
        elseif task == "temporal"
            s = subset_sentence(s, max_seq_len - 1; recent = true, rng = rng)
            masked_word = replace(mask_tokens, :timestamp, 1)
        else
            @assert false
        end
        push!(s, masked_word)
        s
    end    
    
    function get_inputs(sentences, max_seq_len, vocab_sizes, pad_tokens, cls_tokens)
        rng = Random.GLOBAL_RNG
        seq_len = min(maximum(length.(sentences)), max_seq_len)
        tokens =
            get_token_ids(sentences, seq_len, extract(vocab_sizes, :position), pad_tokens, cls_tokens; rng = rng)
        attention_mask = reshape(
            convert.(Float32, extract(tokens, :item) .!= extract(pad_tokens, :item)),
            (1, seq_len, length(sentences)),
        )
        attention_mask = attention_mask .* permutedims(attention_mask, (2, 1, 3))
        tokens, attention_mask
    end;    
    
    function compute_alpha(task)
        checkpoint = sort(parse.(Int, readdir(get_data_path("alphas/$task/Transformer/checkpoints/"))))[end-1]
        params = read_params("$task/Transformer/checkpoints/$checkpoint")
        pretrain_checkpoint = params["pretrain_checkpoint"]
        config = read_params(pretrain_checkpoint)["training_config"]
        model = params["m"]

        s = get_training_data(config["include_ptw_impressions"], config["cls_tokens"])
        s = crop_sentence(s, task, config["max_sequence_length"], config["mask_tokens"])
        tokens, attention_mask = get_inputs(
            [s],
            config["max_sequence_length"],
            config["vocab_sizes"],
            config["pad_tokens"],
            config["cls_tokens"],
        )
        X = model.embed(
            item = extract(tokens, :item),
            rating = extract(tokens, :rating),
            timestamp = extract(tokens, :timestamp),
            status = extract(tokens, :status),
            completion = extract(tokens, :completion),
            position = extract(tokens, :position),
        )
        X = model.transformers(X, attention_mask)
        X = gather(X, [(size(X)[2], 1)])
        item_preds =
            transpose(model.embed.embeddings.item.embedding) *
            model.classifier.item.transform(X) .+ model.classifier.item.output_bias.b
        rating_preds = model.classifier.rating.transform(X)
        item_preds = softmax(item_preds[1:num_items()])
        rating_preds = rating_preds[1:num_items()]

        write_recommendee_alpha(rating_preds, "$task/Transformer/explicit")
        write_recommendee_alpha(item_preds, "$task/Transformer/implicit")
    end    
end

username = ARGS[1]
for task in ALL_TASKS
    compute_alpha(task)
end