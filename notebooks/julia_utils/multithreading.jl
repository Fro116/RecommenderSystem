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

module multithreading

function collect(c::Channel)
    ret = []
    while true
        try
            push!(ret, take!(c))
        catch
            break
        end
    end
    ret
end

end