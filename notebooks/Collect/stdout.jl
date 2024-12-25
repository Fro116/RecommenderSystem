const STDOUT_LOCK = ReentrantLock()

function logtag(tag::AbstractString, x::AbstractString)
    lock(STDOUT_LOCK) do
        println("$(Dates.now()) [$tag] $x")
    end
end

logerror(x::AbstractString) = logtag("ERROR", x)

macro handle_errors(ex)
    quote
        try
            $(esc(ex))
        catch err
            lock(STDOUT_LOCK) do
                Base.showerror(stdout, err, catch_backtrace())
                println()
            end
        end
    end
end
