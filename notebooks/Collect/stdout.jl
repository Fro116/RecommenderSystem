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

macro periodic(tag::AbstractString, secs::Real, expr)
    quote
        while true
            curtime = time()
            sleep($(esc(secs)) * rand() * 0.8)
            logtag($(esc(tag)), "START")
            $(esc(expr))
            sleep_secs = $(esc(secs)) - (time() - curtime)
            logtag($(esc(tag)), "END")
            if sleep_secs < 0
                logtag(tag, "late by $sleep_secs seconds")
            else
                sleep(sleep_secs)
            end
        end
    end
end

macro uniform_delay(secs::Real, expr)
    quote
        sleep($(esc(secs)) * rand())
        $(esc(expr))
    end
end
