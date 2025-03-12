import Logging

Logging.disable_logging(Logging.Info)
READY::Bool = false

function wait_until_ready(port::Integer)
    running = false
    while !running
        try
            ret = HTTP.get("http://localhost:$port/ready"; status_exception=false)
            running = ret.status == 503
        catch e
            sleep(1)
        end
    end
end

Oxygen.@get "/ready" function ready(req::HTTP.Request)
    if !READY
        return HTTP.Response(503, [])
    else
        return HTTP.Response(200, [])
    end
end

Threads.@spawn @handle_errors begin
    logtag("STARTUP", "BEGIN")
    wait_until_ready(PORT)
    logtag("STARTUP", "COMPILING")
    compile(PORT)
    logtag("STARTUP", "END")
    global READY
    READY = true
end

Oxygen.@get "/shutdown" function shutdown(req::HTTP.Request)
    while !READY
        sleep(1)
    end
    Oxygen.terminate()
end

if ! @isdefined MIDDLEWARE
    MIDDLEWARE = []
end
Oxygen.serveparallel(;
    host = "0.0.0.0",
    port = PORT,
    access_log = nothing,
    metrics=false,
    show_banner=false,
    middleware=MIDDLEWARE,
)

