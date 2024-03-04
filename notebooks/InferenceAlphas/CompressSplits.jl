import DataFrames: DataFrame
import Memoize: @memoize

if !@isdefined INFDEF
    INFDEF = true
    import CSV
    import JLD2

    function split_save(file, values)
        Threads.@threads for k in collect(keys(values))
            JLD2.save("$file.$k.jld2", Dict(k => values[k]), compress = true)
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
            data["mediaid"] = Int32.(0:num_items(medium)-1)
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

    function save_dataset(username, source, medium)
        dir = "../../data/recommendations/$source/$username/"
        for split in ["rec_training", "rec_inference"]
            stem = "$dir$medium.$split"
            if split == "rec_training"
                dataset = get_dataset(false, medium, "$dir/user_$(medium)_list.csv")
            elseif split == "rec_inference"
                dataset = get_dataset(true, medium, nothing)
            else
                @assert false
            end
            split_save("$dir/splits/$split.$medium", dataset)
        end
    end

    function runscript(username, source)
        Threads.@threads for medium in ["manga", "anime"]
            save_dataset(username, source, medium)
        end
    end
end

runscript(ARGS...)