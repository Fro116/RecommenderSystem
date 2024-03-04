import NBInclude: @nbinclude
if !@isdefined INFDEF
    IFNDEF = true
    @nbinclude("RecommendationsBase.ipynb")

    function runscript(username, source)
        for m in ALL_MEDIUMS
            df = recommend(m, username, source)
            CSV.write(
                get_data_path("recommendations/$source/$username/recs.$m.csv"),
                df,
            )
        end
    end
end

runscript(ARGS...)