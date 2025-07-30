import CSV
import CSVFiles
import CodecZstd
import Memoize: @memoize

const datadir = "../../../data/import/media"

function read_csv(fn)
    # file contains fields that are too big for the CSV.jl parser
    df = DataFrames.DataFrame(CSVFiles.load(fn, type_detect_rows = 1_000_000))
    if DataFrames.nrow(df) == 0
        return CSV.read(fn, DataFrames.DataFrame) # to get column names
    end
    for col in DataFrames.names(df)
        if eltype(df[!, col]) <: AbstractString
            df[!, col] = replace(df[:, col], "" => missing)
        end
    end
    df
end

function write_csv(fn, df)
    CSV.write(fn, df, bufsize = 2^24)
end

@memoize function get_valid_ids(source::String, medium::String)
    df = read_csv("$datadir/$(source)_$(medium).csv")
    Set(df.itemid)
end

@memoize function get_external(key::String)
    function unquote(x)
        if x[1] == '"' && x[end] == '"'
            return x[2:end-1]
        end
        x
    end
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