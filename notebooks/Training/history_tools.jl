function project_earliest!(user)
    seen = Set()
    items = []
    for x in user["items"]
        tag = x["history_tag"]
        if tag in ["delete"]
            continue
        end
        k = (x["medium"], x["itemid"])
        if k in seen
            continue
        end
        push!(seen, k)
        push!(items, x)
    end
    user["items"] = items
end

function project_latest!(user)
    max_history_tag = ""
    for x in user["items"]
        tag = x["history_tag"]
        if tag in ["infer", "delete"]
            continue
        end
        max_history_tag = max(max_history_tag, tag)
    end
    items = []
    for x in user["items"]
        if x["history_tag"] == max_history_tag
            push!(items, x)
        end
    end
    user["items"] = items
end

function project!(user)
    items = []
    for x in user["items"]
        if x["history_status"] == x["status"] && x["history_rating"] == x["rating"]
            # TODO try not skipping
            continue
        end
        push!(items, x)
    end
    user["items"] = items
end
