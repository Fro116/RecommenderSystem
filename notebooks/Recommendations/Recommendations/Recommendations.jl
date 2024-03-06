import NBInclude: @nbinclude
if !@isdefined INFDEF
    IFNDEF = true
    @nbinclude("RecommendationsBase.ipynb")
end

runscript(ARGS...)