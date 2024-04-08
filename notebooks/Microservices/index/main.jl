import Oxygen
import App

Oxygen.@get "/" App.index
Oxygen.@get "/wake" App.wake
Oxygen.@get "/index" App.index
Oxygen.@post "/heartbeat" App.heartbeat
Oxygen.@post "/submit" App.submit

if length(ARGS) == 0
    port = 8080
    Threads.@spawn begin
        App.precompile(true, port)
        Oxygen.terminate()
    end
elseif length(ARGS) == 1
    port = parse(Int, ARGS[1])
    App.precompile(false, port)
else
    @assert false
end

Oxygen.serveparallel(; host="0.0.0.0", port=port, access_log=nothing)