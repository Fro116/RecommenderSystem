function extract_keywords(document_text::String)::Vector{String}
    pattern = r"(?s)# Keywords\s*\n(.*?)(?=\n\s*#|\z)"
    m = match(pattern, document_text)
    if m === nothing
        return String[]
    end
    keyword_block = m.captures[1]
    keywords = [strip(kw) for kw in split(keyword_block, ',') if !isempty(strip(kw))]
    keywords
end

function extract_queries(document_text::String)::Vector{String}
    pattern = r"(?s)# Queries\s*\n([^#\n].*?)(?=\n[ \t\r]*\n|\n[ \t\r]*#|\z)"
    m = match(pattern, document_text)
    if m === nothing
        return String[]
    end
    keyword_block = m.captures[1]
    function cleanup(x::AbstractString)::String
        m = match(r"^(?:[-*+]|\d+[.)])[ \t]+(.*)$", x)
        isnothing(m) ? "" : strip(m.captures[1])
    end
    queries = []
    for x in split(keyword_block, '\n')
        query = cleanup(x)
        if isempty(query) || length(query) > 256
            if length(x) > 5
                logerror("extract_queries: could not parse $(first(x, 128))")
            end
            continue
        end
        push!(queries, query)
    end
    queries
end

function get_date_tags(data)
    tags = []
    if !isnothing(data[:season])
        push!(tags, replace(data[:season], "-" => " "))
    end
    if !isnothing(data[:startdate]) && !isempty(data[:startdate])
        if length(data[:startdate]) >= 4
            push!(tags, data[:startdate][1:4])
        end
        start_year = parse(Int, data[:startdate][1:4])
        decade_start = floor(Int, start_year / 10) * 10
        if decade_start >= 2000
            push!(tags, "$(decade_start)s")
        else
            push!(tags, "$(decade_start % 100)s")
        end
    end
    tags
end

function get_queries(json)
    tags = []
    if !isnothing(json[:llm_summary])
        append!(tags, extract_keywords(json[:llm_summary][:text]))
        append!(tags, extract_queries(json[:llm_summary][:text]))
    end
    append!(tags, copy(json[:genres]))
    append!(tags, copy(json[:tags]))
    ret = [(x, "tag") for x in tags]
    for k in [:title, :english_title]
        if !isnothing(json[k]) && !isempty(json[k])
            push!(ret, (json[k], "title"))
        end
    end
    for x in get_date_tags(json[:metadata][:dates])
        try
            push!(ret, (x, "date"))
        catch
            println((json[:metadata][:dates]))
        end
    end
    collect(Set([(lowercase(x), y) for (x, y) in ret]))
end