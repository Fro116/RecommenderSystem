import NBInclude: @nbinclude
import Setfield: @set

if !@isdefined TRANSFORMER_IFNDEF
    TRANSFORMER_IFNDEF = true

    source_name = "Transformer"
    import HDF5
    import JSON
    import MLUtils
    import Random
    @nbinclude("../Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Transformer/Data.ipynb")

    function get_training_data(df::RatingsDataset, media, cls_tokens, empty_tokens)
        function itemids(df, i)
            if media[df.source[i]] == "manga"
                return (extract(empty_tokens, :anime), df.item[i])
            elseif media[df.source[i]] == "anime"
                return (df.item[i], extract(empty_tokens, :manga))
            else
                @assert false
            end
        end

        sentences = Dict{Int32,Vector{wordtype}}()
        order = sortperm(df.timestamp)
        for idx = 1:length(order)
            i = order[idx]
            if df.user[i] ∉ keys(sentences)
                sentences[df.user[i]] = [replace(cls_tokens, :user, df.user[i])]
            end
            word = encode_word(
                itemids(df, i)...,
                df.rating[i],
                df.timestamp[i],
                df.status[i],
                df.completion[i],
                df.user[i],
                length(sentences[df.user[i]]),
            )
            push!(sentences[df.user[i]], word)
        end
        sentences
    end

    function get_training_data(task, media, include_ptw, cls_tokens, empty_tokens)
        function get_df(task, content, medium)
            df = get_raw_recommendee_split(content, medium)
            if content != "explicit"
                df.rating .= 11
            end
            df.source .= findfirst(x -> x == df.medium, media)
            Threads.@threads for i = 1:length(df.timestamp)
                # TODO handle timestamps > 1
                df.timestamp[i] = universal_timestamp(df.timestamp[i], df.medium)
            end
            @set df.medium = ""
        end
        contents = ["explicit", "implicit"]
        if include_ptw
            push!(contents, "ptw")
        end
        dfs = [get_df(task, content, medium) for content in contents for medium in media]
        df = reduce(cat, dfs)
        get_training_data(df, media, cls_tokens, empty_tokens)
    end

    function featurize(sentences, task, medium, user, config)
        featurize(;
            sentence = user in keys(sentences) ? sentences[user] :
                       eltype(values(sentences))(),
            task=task,
            medium = medium,
            user = user,
            max_seq_len = config["max_sequence_length"],
            vocab_sizes = config["base_vocab_sizes"],
            pad_tokens = config["pad_tokens"],
            cls_tokens = config["cls_tokens"],
            mask_tokens = config["mask_tokens"],
            empty_tokens = config["empty_tokens"],
        )
    end

    function featurize(;
        sentence::Vector{wordtype},
        task,
        medium,
        user,
        max_seq_len,
        vocab_sizes,
        pad_tokens,
        cls_tokens,
        mask_tokens,
        empty_tokens,
    )
        if length(sentence) == 0
            push!(sentence, replace(cls_tokens, :user, user))
        end
        sentence = subset_sentence(
            sentence,
            min(length(sentence), max_seq_len - 1);
            recent = true,
            keep_first = false,
            rng = nothing,
        )

        # add masking token    
        if task == "temporal"
            masked_word = replace(mask_tokens, :timestamp, 1)
        elseif task == "temporal_causal"
            masked_word = replace(mask_tokens, :timestamp, 1)
            masked_word = replace(masked_word, :position, length(sentence))
        else
            @assert false
        end
        masked_word = replace(masked_word, :user, user)
        push!(sentence, masked_word)
        masked_pos = length(sentence)
        seq_len = max_seq_len

        tokens =
            vec.(
                get_token_ids(
                    [sentence],
                    seq_len,
                    extract(vocab_sizes, :position),
                    pad_tokens,
                    cls_tokens,
                ),
            )
        positions = [masked_pos]

        tokens, positions
    end

    function save_features(sentences, task, medium, config, filename)
        features = [featurize(sentences, task, medium, 1, config)]

        d = Dict{String,AbstractArray}()
        collate = MLUtils.batch
        embed_names = [
            "anime",
            "manga",
            "rating",
            "timestamp",
            "status",
            "completion",
            "user",
            "position",
        ]
        for (i, name) in Iterators.enumerate(embed_names)
            d[name] = collate([x[1][i] for x in features])
        end
        d["positions"] = collate([x[2] for x in features])
        HDF5.h5open(filename, "w") do file
            for (k, v) in d
                write(file, k, v)
            end
        end
    end

    function compute_alpha(username, task, medium, version)
        sourcedir = get_data_path(joinpath("alphas", medium, task, "Transformer", version))
        outdir =
            joinpath(recommendee_alpha_basepath(), medium, task, "Transformer", version)

        f = open(joinpath(sourcedir, "config.json"))
        config = JSON.parse(f)
        close(f)

        sentences = get_training_data(
            task,
            config["media"],
            config["include_ptw_impressions"],
            config["cls_tokens"],
            config["empty_tokens"],
        )

        fn = joinpath(outdir, "inference.h5")
        mkpath(dirname(fn))
        save_features(sentences, task, medium, config, fn)

        run(`python3 Transformer.py --username $username --medium $medium --task $task`)

        file = HDF5.h5open(joinpath(outdir, "embeddings.h5"), "r")
        embeddings = read(file["embedding"])
        item_weight = read(file["$(medium)_item_weight"])'
        item_bias = read(file["$(medium)_item_bias"])
        rating_weight = read(file["$(medium)_rating_weight"])'
        rating_bias = read(file["$(medium)_rating_bias"])
        close(file)

        rating_preds = rating_weight * embeddings[:, 1] + rating_bias
        item_preds = softmax(item_weight * embeddings[:, 1] + item_bias)

        write_recommendee_alpha(
            rating_preds[1:num_items(medium)],
            medium,
            "$medium/$task/Transformer/$version/explicit",
        )

        write_recommendee_alpha(
            item_preds[1:num_items(medium)],
            medium,
            "$medium/$task/Transformer/$version/implicit",
        )
    end
end

username = ARGS[1]
version = "v1"
for medium in ALL_MEDIUMS
    for task in ALL_TASKS
        compute_alpha(username, task, medium, version)
    end
end