import Pkg

dependencies = [
    ("CodecZstd", "0.8.6"),
    ("CSV", "0.10.15"),
    ("DataFrames", "1.7.0"),
    ("HTTP", "1.10.14"),
    ("JSON3", "1.14.1"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("Oxygen", "1.5.15"),
    ("PackageCompiler", "2.2.0"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
