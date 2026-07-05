run(`julia download.jl`)
run(`julia summarize_documents.jl`)
run(`julia embed_documents.jl`)
run(`julia -t 4 embed_queries.jl`) # limit threads to prevent api 429 errors
run(`julia embed_images.jl`)
rm( "../../../data/import/embeddings", force = true, recursive = true)
