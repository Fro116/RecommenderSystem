import Pkg

dependencies = [
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.6"),
    ("CSV", "0.10.15"),
    ("DataFrames", "1.7.0"),
    ("Glob", "1.3.1"),    
    ("HTTP", "1.10.14"),
    ("JLD2", "0.5.10"),
    ("JSON3", "1.14.1"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("NNlib", "0.9.26"),
    ("Oxygen", "1.5.15"),
    ("PackageCompiler", "2.2.0"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
