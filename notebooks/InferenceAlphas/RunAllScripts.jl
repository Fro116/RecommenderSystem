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
            p = run(
                `julia -e 'using DaemonMode; runargs(port)' $script $username $source`,
                wait = false,
            )
        elseif daemon == "false"
            p = run(`julia $script $username $source`, wait = false)
        else
            @assert false
        end
        wait(p)
    end

    function queue(scripts::Vector{String}, port::Int, args...)
        for script in scripts
            execute(script, port, args...)
        end
    end

    function runscript(args...)
        queue(["CompressSplits.jl"], 3001, args...)
        @sync begin
            Threads.@spawn queue(["Baseline.jl", "BagOfWords.jl"], 3002, args...)
            Threads.@spawn queue(["Nondirectional.jl"], 3003, args...)
            Threads.@spawn queue(["Transformer.jl"], 3004, args...)
        end
        queue(["Ensemble.jl"], 3005, args...)
    end
end

runscript(ARGS...)