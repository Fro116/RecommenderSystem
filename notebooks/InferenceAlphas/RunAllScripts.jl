if !@isdefined INFDEF
    IFNDEF = true

    function execute(
        script::String,
        port::Int,
        username::String,
        source::String,
        daemon::String,
    )
        daemon = lowercase(daemon)
        if daemon == "true"
            daemon_mode = "using DaemonMode; runargs($port)"
            p = run(
                `julia --startup-file=no -e $daemon_mode $script $username $source`,
                wait = false,
            )
        elseif daemon == "false"
            p = run(`julia $script $username $source`, wait = false)
        else
            @assert false
        end
        wait(p)
    end

    function queue(scripts::Vector{String}, ports::Vector{Int}, args...)
        for (script, port) in zip(scripts, ports)
            execute(script, port, args...)
        end
    end

    function runscript(args...)
        queue(["CompressSplits.jl"], [3001], args...)
        @sync begin
            Threads.@spawn queue(["Baseline.jl", "BagOfWords.jl"], [3002, 3003], args...)
            Threads.@spawn queue(["Nondirectional.jl"], [3004], args...)
            Threads.@spawn queue(["Transformer.jl"], [3005], args...)
        end
        queue(["Ensemble.jl"], [3006], args...)
    end
end

runscript(ARGS...)