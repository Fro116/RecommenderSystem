import Oxygen
import App

Oxygen.@get "/wake" App.wake
Oxygen.@post "/process" App.process
Oxygen.@post "/compute" App.compute

if length(ARGS) == 0
    port = 8080
    Threads.@spawn begin
        App.precompile(port)
        Oxygen.terminate()
    end
elseif length(ARGS) == 1
    port = parse(Int, ARGS[1])
else
    @assert false
end

Oxygen.serveparallel(; host="0.0.0.0", port=port, access_log=nothing)    

