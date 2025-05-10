import Pkg

dependencies = [
    ("CSV", "0.10.15"),
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.6"),
    ("DataFrames", "1.7.0"),
    ("Glob", "1.3.1"),
    ("HTTP", "1.10.16"),
    ("JSON3", "1.14.2"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("Oxygen", "1.7.1"),
    ("PackageCompiler", "2.2.0"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
