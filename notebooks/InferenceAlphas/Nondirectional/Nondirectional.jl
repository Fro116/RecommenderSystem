#   Nondirectional
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#     •  See the corresponding files in ../../TrainingAlphas for more
#        details

import NBInclude: @nbinclude
import Statistics: mean, var

if !@isdefined NONDIRECTIONAL_IFNDEF
    NONDIRECTIONAL_IFNDEF = true  
    source_name = "Nondirectional"
    @nbinclude("../Alpha.ipynb");
    @nbinclude("../../TrainingAlphas/Explicit/ExplicitItemCFBase.ipynb");   
    
    function compute_related_series_alpha(name)
        df = get_recommendee_split("implicit")
        x = zeros(Float32, num_items())
        x[df.item] .= df.rating
        S = read_similarity_matrix("$name/similarity_matrix")
        preds = S * x
        write_recommendee_alpha(preds, name)
    end
    
    function compute_seen_items_alpha()
        for content in ["explicit", "implicit", "ptw"]
            num_seen::Float32 = length(get_recommendee_split(content).item)
            name = "all/$(uppercasefirst(content))ItemCount"
            write_recommendee_alpha(fill(num_seen, num_items()), name)
        end
    end
    
    function compute_user_stats_alpha()
        r = get_recommendee_split("explicit").rating
        if length(r) == 0
            p_mean = 0f0
        else
            p_mean = mean(r)
        end
        if length(r) <= 1
            p_var = 0f0
        else
            p_var = var(r)
        end
        write_recommendee_alpha(fill(p_mean, num_items()), "all/UserAverage")
        write_recommendee_alpha(fill(p_var, num_items()), "all/UserVariance")
    end    
end

username = ARGS[1]
compute_related_series_alpha("all/RelatedSeries");
compute_related_series_alpha("all/RecapSeries");
compute_seen_items_alpha()
compute_user_stats_alpha()
