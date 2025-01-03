import Dates

STDOUT_LOCK::ReentrantLock = ReentrantLock()

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

macro scheduled(tag::AbstractString, time_str::AbstractString, expr)
    quote
        while true
            now = Dates.now()
            target_time = Dates.DateTime(Dates.format(now, "yyyy-mm-dd") * "T" * $(esc(time_str)), "yyyy-mm-ddTHH:MM:SS")
            if target_time < now
                target_time += Dates.Day(1)
            end
            sleep(target_time - now)
            logtag($(esc(tag)), "START")
            $(esc(expr))
            logtag($(esc(tag)), "END")
        end
    end
end

macro uniform_delay(secs::Real, expr)
    quote
        sleep($(esc(secs)) * rand())
        $(esc(expr))
    end
end

macro timeout(s::Real, f)
    quote
        c = Channel(2)
        Threads.@spawn begin
            sleep($(esc(s)))
            put!(c, :timeout)
        end
        Threads.@spawn begin
            result = $(esc(f))
            put!(c, result)
        end
        take!(c)
    end
end
