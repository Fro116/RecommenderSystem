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
            continue
        end
        push!(items, x)
    end
    user["items"] = items
end

function tokenize!(user)
    function span_to_token(x)
        token = copy(first(x))
        for k in ["status", "rating", "progress"]
            token[k] = last(x)[k]
        end
        token
    end
    items = []
    last_mid = nothing
    span = []
    for x in user["items"]
        mid = (x["medium"], x["matchedid"])
        if mid == last_mid
            push!(span, x)
        else
            if !isempty(span)
                push!(items, span_to_token(span))
            end
            span = [x]
            last_mid = mid
        end
    end
    if !isempty(span)
        push!(items, span_to_token(span))
    end
    user["items"] = items
end
