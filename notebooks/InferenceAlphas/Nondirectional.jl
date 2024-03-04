import NBInclude: @nbinclude
import Memoize: @memoize

if !@isdefined IFNDEF
    IFNDEF = true
    @nbinclude("../TrainingAlphas/Alpha.ipynb")

    @memoize read_similarity_matrix(outdir) = read_params(outdir, true)["S"]

    function write_recommendee_alpha(preds, medium, alpha, username, source)
        model(userids, itemids) = [preds[x+1] for x in itemids]
        write_alpha(model, medium, alpha, REC_SPLITS, username, source)
    end

    function get_recommendee_split(medium, username, source)
        df = get_raw_split(
            "rec_training",
            medium,
            [:itemid, :status],
            nothing,
            username,
            source,
        )
        invalid_statuses = get_status.([:plan_to_watch, :wont_watch, :none])
        filter(df, df.status .∉ (invalid_statuses,))
    end

    function compute_related_series_alpha(df, name, medium, username, source)
        x = zeros(Float32, num_items(medium))
        x[df.itemid.+1] .= 1
        S = read_similarity_matrix(name)
        preds = S * x
        write_recommendee_alpha(preds, medium, name, username, source)
    end

    function compute_cross_related_series_alpha(df, name, medium, username, source)
        if medium == "anime"
            cross_medium = "manga"
        elseif medium == "manga"
            cross_medium = "anime"
        else
            @assert false
        end
        x = zeros(Float32, num_items(cross_medium))
        x[df.itemid.+1] .= 1
        S = read_similarity_matrix(name)
        preds = S' * x
        write_recommendee_alpha(preds, medium, name, username, source)
    end

    function compute_sequel_series_alpha(df, name, medium, username, source)
        x = zeros(Float32, num_items(medium))
        x[df.itemid.+1] .= 1
        S = read_similarity_matrix(name)
        preds = S * x
        write_recommendee_alpha(preds, medium, name, username, source)
    end

    function compute_dependency_alpha(df, name, outdir, medium, username, source)
        df = filter(df, df.status .>= get_status(:completed))
        x = ones(Float32, num_items(medium))
        x[df.itemid.+1] .= 0
        S = read_similarity_matrix(name)
        preds = S * x
        write_recommendee_alpha(preds, medium, outdir, username, source)
    end

    function runscript(username, source)
        anime_df = get_recommendee_split("anime", username, source)
        manga_df = get_recommendee_split("manga", username, source)
        @sync for medium in ALL_MEDIUMS
            if medium == "anime"
                media_df = anime_df
                cross_media_df = manga_df
            elseif medium == "manga"
                media_df = manga_df
                cross_media_df = anime_df
            else
                @assert false
            end
            Threads.@spawn compute_related_series_alpha(
                media_df,
                "$medium/Nondirectional/RelatedSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_related_series_alpha(
                media_df,
                "$medium/Nondirectional/RecapSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_cross_related_series_alpha(
                cross_media_df,
                "$medium/Nondirectional/CrossRelatedSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_cross_related_series_alpha(
                cross_media_df,
                "$medium/Nondirectional/CrossRecapSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_sequel_series_alpha(
                media_df,
                "$medium/Nondirectional/SequelSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_sequel_series_alpha(
                media_df,
                "$medium/Nondirectional/DirectSequelSeries",
                medium,
                username,
                source,
            )
            Threads.@spawn compute_dependency_alpha(
                media_df,
                "$medium/Nondirectional/SequelSeries",
                "$medium/Nondirectional/Dependencies",
                medium,
                username,
                source,
            )
        end
    end
end

runscript(ARGS...)