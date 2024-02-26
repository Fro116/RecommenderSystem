import NBInclude: @nbinclude

if !@isdefined NONDIRECTIONAL_IFNDEF
    NONDIRECTIONAL_IFNDEF = true
    @nbinclude("../../TrainingAlphas/Alpha.ipynb")

    function read_similarity_matrix(outdir)
        read_params(outdir, true)["S"]
    end

    function  write_recommendee_alpha(preds, medium, name)
        model(userids, itemids) = [preds[x+1] for x in itemids]
        write_alpha(model, medium, name, REC_SPLITS)        
    end

    function get_recommendee_split(medium)
        get_split(
            "rec_training",
            "watch",
            medium,
            [:itemid, :status],
        )        
    end

    function compute_related_series_alpha(name, medium)
        df = get_recommendee_split(medium)
        x = zeros(Float32, num_items(medium))
        x[df.itemid .+ 1] .= 1
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
        df = get_recommendee_split(cross_medium)
        x = zeros(Float32, num_items(cross_medium))
        x[df.itemid .+ 1] .= 1
        S = read_params(name, true)["S"]
        preds = S' * x
        write_recommendee_alpha(preds, medium, name)
    end

    function compute_sequel_series_alpha(name, medium)
        df = get_recommendee_split(medium)
        df = filter(df, df.status .>= get_status(:completed))
        x = zeros(Float32, num_items(medium))
        x[df.itemid .+ 1] .= 1
        S = read_similarity_matrix("$name/similarity_matrix")
        preds = S * x
        write_recommendee_alpha(preds, medium, name)
    end
    
    function compute_dependency_alpha(name, outdir, medium)
        df = get_recommendee_split(medium)
        df = filter(df, df.status .>= get_status(:completed))
        x = ones(Float32, num_items(medium))
        x[df.itemid] .= 0
        S = read_similarity_matrix("$name/similarity_matrix")
        preds = S * x
        write_recommendee_alpha(preds, medium, outdir)
    end    
end

username = ARGS[1]
source = ARGS[2]
for medium in ALL_MEDIUMS
    compute_related_series_alpha("$medium/Nondirectional/RelatedSeries", medium)
    compute_related_series_alpha("$medium/Nondirectional/RecapSeries", medium)
    compute_cross_related_series_alpha("$medium/Nondirectional/CrossRelatedSeries", medium)
    compute_cross_related_series_alpha("$medium/Nondirectional/CrossRecapSeries", medium)   
    compute_sequel_series_alpha("$medium/Nondirectional/SequelSeries", medium)
    compute_sequel_series_alpha("$medium/Nondirectional/DirectSequelSeries", medium)    
    compute_dependency_alpha("$medium/Nondirectional/SequelSeries", "$medium/Nondirectional/Dependencies", medium)    
end
