import NBInclude: @nbinclude
if !@isdefined TRANSFORMER_IFNDEF
    TRANSFORMER_IFNDEF = true

    import H5Zblosc
    import HDF5
    import JSON
    import MLUtils
    import NNlib: sigmoid
    import Random
    @nbinclude("../../TrainingAlphas/Alpha.ipynb")
    @nbinclude("../../TrainingAlphas/Transformer/Data.ipynb")

    function get_training_data(df::RatingsDataset, cls_tokens, max_seq_length)
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

    function get_training_data(
        cls_tokens,
        max_seq_length,
        target_medium::String,
        exclude_ptw::Bool,
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
            df = get_raw_split("rec_training", medium, fields, nothing)
            if exclude_ptw && medium == target_medium
                df = filter(df, df.status .!= get_status(:plan_to_watch))
            end
            df
        end
        dfs = [get_df(medium) for medium in ALL_MEDIUMS]
        df = reduce(cat, dfs)
        get_training_data(df, cls_tokens, max_seq_length)
    end

    function tokenize(sentence, medium, config)
        tokenize(;
            sentence = copy(sentence),
            medium = medium,
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
        medium,
        userid,
        max_seq_len,
        vocab_sizes,
        pad_tokens,
        cls_tokens,
        mask_tokens,
    )
        # get inputs
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

    function save_tokens(sentences, medium, config, filename)
        tokens = [tokenize(x, medium, config) for x in sentences]
        d = Dict{String,AbstractArray}()
        collate = MLUtils.batch
        for (i, name) in Iterators.enumerate(config.vocab_names)
            d[name] = collate([x[1][i] for x in tokens])
        end
        d["positions"] = collate([x[2] for x in tokens])
        HDF5.h5open(filename, "w") do f
            for (k, v) in d
                f[k, blosc = 3] = v
            end
        end
    end

    function compute_alpha(source, username, medium, version)
        outdir = joinpath(
            get_data_path("recommendations/$(get_rec_usertag())"),
            "alphas",
            medium,
            "Transformer",
            version,
        )
        mkpath(outdir)

        sourcedir = get_data_path(joinpath("alphas", "all", "Transformer", version, "0"))
        f = open(joinpath(sourcedir, "config.json"))
        d = JSON.parse(f)
        config = NamedTuple(Symbol.(keys(d)) .=> values(d))
        close(f)

        sentences = [
            get_training_data(
                config[:cls_tokens],
                config[:max_sequence_length],
                medium,
                include_ptw,
            ) for include_ptw in [false, true]
        ]
        save_tokens(sentences, medium, config, "$outdir/inference.h5")
        run(`python3 Transformer.py --source $source --username $username --medium $medium`)
        file = HDF5.h5open(joinpath(outdir, "embeddings.h5"), "r")
        seen = get_raw_split("rec_training", medium, [:itemid], nothing).itemid
        ptw = get_split("rec_training", "plantowatch", medium, [:itemid], nothing).itemid
        watched = [x for x in seen if x ∉ Set(ptw)]
        for metric in ALL_METRICS
            M = read(file["$(medium)_$(metric)"])
            r = M[:, 1] # regular items
            p = M[:, 2] # plantowatch items
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
            model(userids, itemids) = [e[x+1] for x in itemids]
            write_alpha(model, medium, "$medium/Transformer/$version/$metric", REC_SPLITS)
        end
        close(file)
    end
end

username = ARGS[1]
source = ARGS[2]
version = "v1"
for medium in ALL_MEDIUMS
    compute_alpha(source, username, medium, version)
end