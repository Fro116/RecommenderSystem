if !@isdefined COMPRESS_SPLITS_IFNDEF
    COMPRESS_SPLITS_IFNDEF = true
    using CSV
    using DataFrames
    using JLD2    
    struct RatingsDataset
        user::Vector{Int32}
        item::Vector{Int32}
        rating::Vector{Float32}
        timestamp::Vector{Float32}
        status::Vector{Int32}
        completion::Vector{Float32}
        rewatch::Vector{Int32}
        source::Vector{Int32}
    end    
    
    function get_dataset(file)
        df = DataFrame(CSV.File(file))
        RatingsDataset(
            fill(1, length(df.username)), # julia is 1 indexed
            df.animeid .+ 1, # julia is 1 indexed
            df.score,
            df.timestamp,
            df.status,
            df.completion,
            df.rewatch,
            df.source,
        )
    end    
    
    dir = "../../data/recommendations"    
end;

username = ARGS[1]
for content in ["explicit", "implicit", "ptw"]
    stem = "$dir/$username/$content"
    dataset = get_dataset("$stem.csv")
    jldsave("$stem.jld2"; dataset)
end