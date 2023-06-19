import NBInclude: @nbinclude

if !@isdefined NONDIRECTIONAL_IFNDEF
    NONDIRECTIONAL_IFNDEF = true
    source_name = "Nondirectional"
    import Statistics: mean, var
    @nbinclude("../Alpha.ipynb")

    function read_similarity_matrix(outdir)
        read_params(outdir)["S"]
    end

    function compute_related_series_alpha(name, medium)
        df = get_recommendee_split("implicit", medium)
        x = zeros(Float32, num_items(medium))
        x[df.item] .= 1
        S = read_similarity_matrix("$name/similarity_matrix")
        preds = S * x
        write_recommendee_alpha(preds, medium, name)
    end

    function compute_cross_related_series_alpha(name, medium)
        if medium == "anime"
            cross_medium = "manga"
        elseif medium == "manga"
            cross_medium = "anime"
        else
            @assert false
        end
        df = get_recommendee_split("implicit", cross_medium)
        x = zeros(Float32, num_items(cross_medium))
        x[df.item] .= 1
        S = read_params(name)["S"]
        preds = S' * x
        write_recommendee_alpha(preds, medium, name)
    end

    function compute_sequel_series_alpha(name, medium)
        df = get_recommendee_split("implicit", medium)
        df = filter(df, df.status .== 5)
        x = ones(Float32, num_items(medium))
        x[df.item] .= 0
        S = read_similarity_matrix("$name/similarity_matrix")
        preds = S * x
        write_recommendee_alpha(preds, medium, name)
    end

    function compute_seen_items_alpha(medium)
        for content in ["explicit", "implicit", "ptw"]
            num_seen::Float32 = length(get_recommendee_split(content, medium).item)
            name = "$medium/all/$(uppercasefirst(content))ItemCount"
            write_recommendee_alpha(fill(num_seen, num_items(medium)), medium, name)
        end
    end

    function compute_user_stats_alpha(medium)
        r = get_recommendee_split("explicit", medium).rating
        if length(r) == 0
            p_mean = 0.0f0
        else
            p_mean = mean(r)
        end
        if length(r) <= 1
            p_var = 0.0f0
        else
            p_var = var(r)
        end
        write_recommendee_alpha(
            fill(p_mean, num_items(medium)),
            medium,
            "$medium/all/UserAverage",
        )
        write_recommendee_alpha(
            fill(p_var, num_items(medium)),
            medium,
            "$medium/all/UserVariance",
        )
    end
end

username = ARGS[1]
for medium in ALL_MEDIUMS
    compute_related_series_alpha("$medium/all/RelatedSeries", medium)
    compute_related_series_alpha("$medium/all/RecapSeries", medium)
    compute_cross_related_series_alpha("$medium/all/CrossRelatedSeries", medium)
    compute_sequel_series_alpha("$medium/all/SequelSeries", medium)
    compute_seen_items_alpha(medium)
    compute_user_stats_alpha(medium)
end
