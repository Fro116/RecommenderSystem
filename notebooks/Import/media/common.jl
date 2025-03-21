import CodecZstd
import Memoize: @memoize

const datadir = "../../../data/import/media"

@memoize function get_valid_ids(source::String, medium::String)
    df = CSV.read("$datadir/$(source)_$(medium).csv", DataFrames.DataFrame, ntasks=1)
    Set(df.itemid)
end

@memoize function get_external(key::String)
    function unquote(x)
        if x[1] == '"' && x[end] == '"'
            return x[2:end-1]
        end
        x
    end
    # file contains fields that are too big for the CSV.jl parser
    lines = readlines("$datadir/external/external_dependencies.csv")
    headers = split(lines[1], ",")
    key_col = findfirst(==("key"), headers)
    value_col = findfirst(==("value"), headers)
    for line in lines
        fields = unquote.(split(line, ","))
        if fields[key_col] == key
            value = fields[value_col]
            bytes = hex2bytes(value[3:end])
            return String(CodecZstd.transcode(CodecZstd.ZstdDecompressor, bytes))
        end
    end
    @assert false
end