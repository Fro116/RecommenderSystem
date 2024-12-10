function writelogs(filename, num_files, lines_per_file)
    file_index = 0
    lines = 0
    mode = "w"
    while true
        line = readline()
        open("$filename.$file_index", mode) do f
            write(f, line * "\n")
            flush(f)
        end
        lines += 1
        mode = "a"
        if lines == lines_per_file
            lines = 0
            file_index = (file_index + 1) % num_files
            mode = "w"
        end
    end
end

writelogs(ARGS[1], 1, 10000)
