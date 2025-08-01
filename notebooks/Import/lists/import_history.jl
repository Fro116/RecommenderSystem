module histories

include("../../julia_utils/stdout.jl")

import Dates

const min_ts = Dates.datetime2unix(Dates.DateTime(2000, 1, 1))
const ts_epsilon = 1

function to_valid_ts(x, min_ts::Float64, max_ts::Float64)::Union{Nothing, Float64}
    if isnothing(x) || x < min_ts || x > max_ts + 86400 # account for timezones
        return nothing
    end
    Float64(x)
end

function to_mal_time(x, min_ts::Float64, max_ts::Float64)::Union{Nothing, Float64}
    if isnothing(x)
        return nothing
    end
    for f in ["yyyy-mm-dd", "yyyy-mm-ddTHH:MM:SS+00:00"]
        if length(x) == length(f)
            ts = Dates.datetime2unix(Dates.DateTime(x, f))
            return to_valid_ts(ts, min_ts, max_ts)
        end
    end
    nothing
end

function to_anilist_time(x, min_ts::Float64, max_ts::Float64)::Union{Nothing, Float64}
    to_valid_ts(x, min_ts, max_ts)
end

function to_kitsu_time(x, min_ts::Float64, max_ts::Float64)::Union{Nothing, Float64}
    if isnothing(x)
        return nothing
    end
    ts = Dates.datetime2unix(Dates.DateTime(x[1:end-1], "yyyy-mm-ddTHH:MM:SS.sss"))
    to_valid_ts(ts, min_ts, max_ts)
end

function to_animeplanet_time(x, min_ts::Float64, max_ts::Float64)::Union{Nothing, Float64}
    if isnothing(x)
        return nothing
    end
    to_valid_ts(x, min_ts, max_ts)
end

function get_col(x, source::AbstractString, key::AbstractString, max_ts::Union{Float64, Nothing})
    if source == "mal"
        if key == "created_at"
            return to_mal_time(x["start_date"], min_ts, max_ts)
        elseif key == "updated_at"
            return to_mal_time(x["updated_at"], min_ts, max_ts)
        elseif key == "rating"
            return x["score"]
        elseif key == "status"
            return x["status"]
        elseif key == "progress"
            return x["progress"]
        elseif key == "progress_volumes"
            return x["num_volumes_read"]
        end
    elseif source == "anilist"
        if key == "created_at"
            return to_anilist_time(x["createdat"], min_ts, max_ts)
        elseif key == "updated_at"
            return to_anilist_time(x["updatedat"], min_ts, max_ts)
        elseif key == "rating"
            return x["score"]
        elseif key == "status"
            return x["status"]
        elseif key == "progress"
            return x["progress"]
        elseif key == "progress_volumes"
            return x["progressvolumes"]
        end
    elseif source == "kitsu"
        if key == "created_at"
            return to_kitsu_time(x["createdat"], min_ts, max_ts)
        elseif key == "updated_at"
            return to_kitsu_time(x["updatedat"], min_ts, max_ts)
        elseif key == "rating"
            return x["rating"]
        elseif key == "status"
            return x["status"]
        elseif key == "progress"
            return x["progress"]
        elseif key == "progress_volumes"
            return x["progress"]
        end
    elseif source == "animeplanet"
        if key == "created_at"
            return nothing
        elseif key == "updated_at"
            return to_animeplanet_time(x["updated_at"], min_ts, max_ts)
        elseif key == "rating"
            return x["score"]
        elseif key == "status"
            return x["status"]
        elseif key == "progress"
            return x["progress"]
        elseif key == "progress_volumes"
            return x["progress"]
        end
    end
    @assert false "$source $key"
end

function infer_history(user, source::AbstractString, db_refreshed_at::Float64)
    items = []
    for x in user["items"]
        created_at = get_col(x, source, "created_at", db_refreshed_at)
        if isnothing(created_at)
            continue
        end
        item = copy(x)
        item["history_min_ts"] = created_at
        item["history_max_ts"] = created_at
        item["history_tag"] = "infer"
        push!(items, item)
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    user = copy(user)
    user["items"] = items
    user["user"]["history_ts"] = min_ts
    user
end

function is_consistent_item(x, y, source::AbstractString)::Bool
    get_col(x, source, "rating", nothing) == get_col(y, source, "rating", nothing) &&
        get_col(x, source, "status", nothing) == get_col(y, source, "status", nothing) &&
        get_col(x, source, "progress", nothing) == get_col(y, source, "progress", nothing) &&
        get_col(x, source, "progress_volumes", nothing) == get_col(y, source, "progress_volumes", nothing) &&
        x["history_min_ts"] < y["history_max_ts"] + ts_epsilon/2 &&
        y["history_min_ts"] < x["history_max_ts"] + ts_epsilon/2
end

function is_duplicate_item(x, y, source::AbstractString)::Bool
    for k in intersect(keys(x), keys(y))
        if startswith(k, "history") || k in ["item_order"]
            continue
        end
        if x[k] != y[k]
            return false
        end
    end
    true
end

function deduplicate_user(user, source::AbstractString)
    user = copy(user)
    if source == "anilist"
        statuses =
            [nothing, "DROPPED", "PAUSED", "PLANNING", "CURRENT", "COMPLETED", "REPEATING"]
        statuses = Dict(x => i for (i, x) in Iterators.enumerate(statuses))
        seen = Dict()
        for (i, x) in Iterators.enumerate(user["items"])
            k = (x["medium"], x["itemid"])
            v = (i, get(statuses, x["status"], 0))
            if k ∉ keys(seen)
                seen[k] = v
            else
                if v[2] > seen[k][2]
                    seen[k] = v
                end
            end
        end
        keep = Set(v[1] for v in values(seen))
        items = [x for (i, x) in Iterators.enumerate(user["items"]) if i in keep]
        user["items"] = items
    else
        seen = Set()
        skipped = false
        items = []
        for x in user["items"]
            k = (x["medium"], x["itemid"])
            if k ∉ seen
                push!(seen, k)
                push!(items, x)
            else
                skipped = true
            end
        end
        user["items"] = items
        if skipped
            user_str =
                get(user["user"], "username", "") *
                " " *
                string(get(user["user"], "userid", ""))
            logerror("skipping duplicate items for $source $user_str")
        end
    end
    user
end

function update_history(
    hist,
    user,
    source::AbstractString,
    db_refreshed_at::Float64,
    datetag::AbstractString,
)
    user = deduplicate_user(user, source)
    if isnothing(hist)
        hist = infer_history(user, source, db_refreshed_at)
    end
    
    # add timestamp ranges to new list
    items = user["items"]
    if source == "animeplanet"
        # old items have null item_orders
        for m in ["manga", "anime"]
            item_order = 0
            for x in items
                if x["medium"] != m
                    continue
                end
                if isnothing(x["item_order"])
                    x["item_order"] = item_order
                end
                item_order += 1
            end
        end
        # set max_ts
        sort!(items, by = x -> x["item_order"])
        for m in ["manga", "anime"]
            ts_upper_bound = db_refreshed_at
            for x in items
                if x["medium"] != m
                    continue
                end
                ts = get_col(x, source, "updated_at", db_refreshed_at)
                if !isnothing(ts)
                    ts_upper_bound = ts
                end
                x["history_max_ts"] = ts_upper_bound
                ts_upper_bound -= ts_epsilon
            end
        end
        # set min_ts
        sort!(items, by = x -> x["item_order"], rev=true)
        for m in ["manga", "anime"]
            ts_lower_bound = min_ts
            for x in items
                if x["medium"] != m
                    continue
                end
                ts = get_col(x, source, "updated_at", db_refreshed_at)
                if !isnothing(ts)
                    ts_lower_bound = ts
                end
                x["history_min_ts"] = ts_lower_bound
                ts_lower_bound += ts_epsilon
            end
        end
        # edge cases
        for x in items
            if x["history_min_ts"] > x["history_max_ts"]
                x["history_min_ts"] = x["history_max_ts"]
            end
        end
    else
        for x in items
            ts = get_col(x, source, "updated_at", db_refreshed_at)
            if !isnothing(ts)
                x["history_min_ts"] = ts
                x["history_max_ts"] = ts
            else
                x["history_min_ts"] = min_ts
                x["history_max_ts"] = db_refreshed_at
            end
        end
    end
    sort!(items, by = x -> (x["history_max_ts"], x["history_min_ts"]))

    # merge new list into the history
    snapshots = Dict()
    prev_snapshot = Dict()
    for x in hist["items"]
        k = (x["medium"], x["itemid"])
        if k ∉ keys(snapshots)
            snapshots[k] = []
        end
        push!(snapshots[k], x)
        if x["history_tag"] == "delete"
            delete!(prev_snapshot, k)
        else
            prev_snapshot[k] = x
        end
    end
    for x in items
        k = (x["medium"], x["itemid"])
        if k ∉ keys(snapshots)
            snapshots[k] = []
        end
        duplicate_item = false
        for y in snapshots[k]
            if y["history_tag"] ∉ ["infer", "delete"] && is_duplicate_item(x, y, source)
                y["history_tag"] = datetag
                duplicate_item = true
                break
            end
        end
        if duplicate_item
            continue
        end
        if k ∉ keys(prev_snapshot)
            x["history_min_ts"] = max(x["history_min_ts"], hist["user"]["history_ts"])
            x["history_tag"] = datetag
            push!(snapshots[k], x)
        else
            y = prev_snapshot[k]
            if is_consistent_item(x, y, source)
                x["history_min_ts"] = max(x["history_min_ts"], y["history_min_ts"])
                x["history_max_ts"] = min(x["history_max_ts"], y["history_max_ts"])
                y["history_max_ts"] = Inf # set to Inf to remove y
                y["history_min_ts"] = Inf
            else
                x["history_min_ts"] = max(x["history_min_ts"], hist["user"]["history_ts"])
            end
            x["history_tag"] = datetag
            push!(snapshots[k], x)
        end
    end
    for k in collect(keys(snapshots))
        v = snapshots[k]
        sort!(v, by = x -> (x["history_max_ts"], x["history_min_ts"]))
        idx = findfirst(==(datetag), [x["history_tag"] for x in v])
        if !isnothing(idx)
            v = v[1:idx]
        end
        snapshots[k] = v
    end
    merged_items = []
    for v in values(snapshots)
        append!(merged_items, v)
    end

    # delete items that are not in the new list
    new_items = Set((x["medium"], x["itemid"]) for x in items)
    for (k, x) in prev_snapshot
        if k ∉ new_items
            item = copy(x)
            item["history_min_ts"] = hist["user"]["history_ts"]
            item["history_max_ts"] = db_refreshed_at
            item["history_tag"] = "delete"
            push!(merged_items, item)
        end
    end

    # return
    for x in merged_items
        if x["history_min_ts"] > x["history_max_ts"]
            x["history_min_ts"] = x["history_max_ts"]
        end
    end
    sort!(merged_items, by = x -> (x["history_max_ts"], x["history_min_ts"]))
    newhist = copy(user)
    newhist["user"]["history_ts"] = db_refreshed_at
    newhist["items"] = merged_items
    newhist
end

end