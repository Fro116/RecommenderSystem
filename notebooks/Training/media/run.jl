for fn in [
    "save_media.jl",
    "match_ids.jl",
    "match_manami.jl",
    "match_manual.jl",
    "match_metadata.jl",
]
    run(`julia $fn`)
end
