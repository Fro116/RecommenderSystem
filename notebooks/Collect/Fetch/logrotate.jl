function writelogs(filename, lines_per_file)
    lines = 0
    mode = "w"
    while true
        line = readline()
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

writelogs(ARGS[1], 10000)
