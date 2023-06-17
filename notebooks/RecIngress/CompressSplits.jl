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
        source::Vector{Int32}
        medium::String        
    end    
    
    function get_dataset(file, medium)
        df = DataFrame(CSV.File(file))
        RatingsDataset(
            fill(1, length(df.username)), # julia is 1 indexed
            df[:, "$(medium)id"] .+ 1, # julia is 1 indexed
            df.score,
            df.timestamp,
            df.status,
            df.completion,
            df.source,
            medium
        )
    end    
    
    dir = "../../data/recommendations"    
end;

username = ARGS[1]
for medium in ["anime", "manga"]
    for content in ["explicit", "implicit", "ptw"]
        stem = "$dir/$username/$medium.$content"
        dataset = get_dataset("$stem.csv", medium)
        jldsave("$stem.jld2"; dataset)
    end
end