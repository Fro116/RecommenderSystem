import CodecZstd
import CSV
import DataFrames
import HTTP
import JSON3
import MsgPack
import ProgressMeter: @showprogress

include("../../julia_utils/stdout.jl")

const datadir = "../../../data/import/embeddings/documents"
const mediums = ["manga", "anime"]
const sources = ["mal", "anilist", "kitsu", "animeplanet"]

function download_data()
    logtag("FORMAT_DOCUMENTS", "downloading data")
    rm(datadir, force = true, recursive = true)
    mkpath(datadir)
    retrieval = "rclone --retries=10 copyto r2:rsys/database/import"
    files = vcat(
        ["$m.groups.csv" for m in mediums],
        ["$(s)_$(m).json" for s in sources for m in mediums],
        ["embeddings.json"],
    )
    for fn in files
        cmd = "$retrieval/$fn $datadir/$fn"
        run(`sh -c $cmd`)
    end
end

function get_media(source::String, medium::String)
    fn = "$datadir/$(source)_$(medium).json"
    df = copy(JSON3.read(fn))
    for x in df
        x[:itemsource] = source
    end
    df
end

get_media(medium::String) = reduce(vcat, [get_media(s, medium) for s in sources])

function get_media_groups(medium::AbstractString)
    fn = "$datadir/$medium.groups.csv"
    groups =
        CSV.read(fn, DataFrames.DataFrame, types = Dict("itemid" => String), ntasks = 1)
    groups = Dict((x.source, x.itemid) => x for x in eachrow(groups))
    media = get_media(medium)
    for x in media
        k = (x[:itemsource], x[:itemid])
        v = groups[k]
        x[:groupid] = v.groupid
        x[:count] = v.count
        x[:distinctid] = 0
        x[:matchedid] = 0
    end
    sort!(media, by = x -> x[:count], rev = true)
    min_count = 100
    distinctid = 0
    groupmap = Dict()
    for x in media
        if x[:count] < min_count
            x[:distinctid] = 0
            x[:matchedid] = get(groupmap, x[:groupid], 0)
        else
            distinctid += 1
            if x[:groupid] ∉ keys(groupmap)
                groupmap[x[:groupid]] = length(groupmap) + 1
            end
            x[:distinctid] = distinctid
            x[:matchedid] = groupmap[x[:groupid]]
        end
    end
    media
end

function create_document(item, matchedids)
    function recommendation(x)
        if ismissing(x)
            return missing
        end
        k = (item[:medium], item[:itemsource], x[:itemid])
        y = copy(x)
        delete!(y, :itemid)
        y[:matchedid] = get(matchedids, k, 0)
        y[:itemsource] = item[:itemsource]
        y
    end
    Dict(
        :title => item[:title],
        :english_title => item[:english_title],
        :metadata => Dict(
            :medium => item[:medium],
            :mediatype => item[:mediatype],
            :dates => Dict(
                :startdate => item[:startdate],
                :enddate => item[:enddate],
                :season => item[:season],
            ),
            :length => Dict(
                :episodes => item[:episodes],
                :duration => item[:duration],
                :volumes => item[:volumes],
                :chapters => item[:chapters],
            ),
            :status => item[:status],
            :source_material => item[:source],
            :authors => item[:authors],
            :studios => item[:studios],
            :alternative_titles => item[:alternative_titles],
        ),
        :synopsis => [item[:synopsis]],
        :characters => item[:characters],
        :genres => item[:genres],
        :tags => item[:tags],
        :background => [item[:background]],
        :reviews => item[:reviews],
        :recommendations => recommendation.(item[:recommendations]),
        :keys => [(item[:medium], item[:itemsource], item[:itemid])],
        :count => Dict(item[:itemsource] => item[:count]),
    )
end

function merge_documents(x, y)
    function merge_entry(a, b)
        if ismissing(a)
            return b
        end
        if ismissing(b)
            return a
        end
        vcat(a, b)
    end
    ret = copy(x)
    for k in
        [:synopsis, :characters, :genres, :tags, :background, :reviews, :recommendations, :keys]
        ret[k] = merge_entry(x[k], y[k])
    end
    for k in keys(y[:count])
        if k ∉ keys(ret[:count])
            ret[:count][k] = y[:count][k]
        end
    end
    ret
end

function get_documents(items)
    matchedids =
        Dict((x[:medium], x[:itemsource], x[:itemid]) => x[:matchedid] for x in items)
    documents = Dict()
    for x in items
        if x[:matchedid] == 0
            continue
        end
        k = x[:matchedid]
        d = create_document(x, matchedids)
        if k ∉ keys(documents)
            documents[k] = d
        else
            documents[k] = merge_documents(documents[k], d)
        end
    end
    documents
end

function format_documents!(documents)
    for (_, v) in documents
        sort!(v[:reviews], by = r -> r[:count], rev = true)
        recs = Dict()
        for r in v[:recommendations]
            k = r[:matchedid]
            if k ∉ keys(documents)
                continue
            end
            if k ∉ keys(recs)
                recs[k] = Dict(
                    :title => documents[k][:title],
                    :matchedid => k,
                    :synopsis => first(documents[k][:synopsis], 1),
                    :count => Dict(),
                    :reasons => [],
                )
            end
            recs[k][:count][r[:itemsource]] = max(get(recs[k][:count], r[:itemsource], 0), r[:count])
            if !isempty(r[:text])
                push!(recs[k][:reasons], r[:text])
            end
        end
        for x in values(recs)
            x[:count] = isempty(x[:count]) ? 0 : sum(values(x[:count]))
        end
        v[:recommendations] = sort(collect(values(recs)), by = x -> x[:count], rev = true)
    end
end

download_data()
for m in mediums
    items = get_media_groups(m)
    documents = get_documents(items)
    format_documents!(documents)
    documents = Dict(string(k) => v for (k, v) in documents)
    open("$datadir/$m.json", "w") do f
        JSON3.write(f, documents)
    end
end
