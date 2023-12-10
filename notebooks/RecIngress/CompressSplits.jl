using CSV
using DataFrames
using JLD2
using Memoize

if !@isdefined COMPRESS_SPLITS_IFNDEF
    COMPRESS_SPLITS_IFNDEF = true

    function split_save(file, values)
        # save in multiple files to allow multithreaded reading
        for (k, v) in values
            JLD2.save("$file.$k.jld2", Dict(k => v), compress = true)
        end
    end

    @memoize function num_items(medium)
        df = DataFrame(CSV.File("../../data/processed_data/$(medium)_to_uid.csv"))
        length(df.uid)
    end

    function get_dataset(inference, medium, file)
        data = Dict(
            "source" => Int32[],
            "medium" => Int32[],
            "userid" => Int32[],
            "mediaid" => Int32[],
            "status" => Int32[],
            "rating" => Float32[],
            "backward_order" => Int32[],
            "forward_order" => Int32[],
            "updated_at" => Float32[],
            "created_at" => Float32[],
            "started_at" => Float32[],
            "finished_at" => Float32[],
            "progress" => Float32[],
            "repeat_count" => Int32[],
            "priority" => Float32[],
            "sentiment" => Int32[],
            "sentiment_score" => Float32[],
            "owned" => Float32[],
        )

        @assert inference == isnothing(file)
        if inference
            data["userid"] = Int32[0 for _ = 1:num_items(medium)]
            data["mediaid"] = Int32.(1:num_items(medium))
        else
            type_parser =
                (_, name) ->
                    String(name) in keys(data) ? eltype(data[String(name)]) : Float32
            # load in chunks to reduce memory usage
            df = DataFrame(CSV.File(file, types = type_parser))
            for k in keys(data)
                append!(data[k], df[:, k])
            end
        end

        # rename columns
        data["itemid"] = data["mediaid"]
        data["update_order"] = data["backward_order"]
        delete!(data, "mediaid")
        delete!(data, "backward_order")
        delete!(data, "forward_order")
        data
    end

    function save_dataset(username, medium)
        dir = "../../data/recommendations/$username/"
        for split in ["rec_training", "rec_inference"]
            stem = "$dir$medium.$split"
            if split == "rec_training"
                dataset =
                    get_dataset(false, medium, "$dir/user_$(medium)_list.csv")
            elseif split == "rec_inference"
                dataset = get_dataset(true, medium, nothing)
            else
                @assert false
            end
            split_save("$dir/splits/$split.$medium", dataset)
        end
    end
end

username = ARGS[1]
for medium in ["manga", "anime"]
    save_dataset(username, medium)
end