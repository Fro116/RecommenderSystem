function writelogs(filename, lines_per_file)
    lines = 0
    mode = "w"
    while true
        line = readline()
        if isempty(line)
            continue
        end
        open(filename, mode) do f
            write(f, line * "\n")
            flush(f)
        end
        lines += 1
        mode = "a"
        if lines == lines_per_file
            lines = 0
            mode = "w"
        end
    end
end

writelogs(ARGS[1], get(ARGS, 2, 1_000_000))
