import Pkg

dependencies = [
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.7"),
    ("CSV", "0.10.16"),
    ("DataFrames", "1.8.2"),
    ("Glob", "1.4.0"),
    ("HTTP", "1.11.0"),
    ("JLD2", "0.6.4"),
    ("JSON3", "1.14.3"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("NNlib", "0.9.34"),
    ("Oxygen", "1.10.2"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
