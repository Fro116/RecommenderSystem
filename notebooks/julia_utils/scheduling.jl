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
                logtag($(esc(tag)), "late by $sleep_secs seconds")
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
            while Dates.now() < target_time
                sleep(max(target_time - Dates.now(), Dates.Second(0)))
            end
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
