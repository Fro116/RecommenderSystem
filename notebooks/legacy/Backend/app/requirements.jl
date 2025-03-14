import Pkg

dependencies = [
    ("CodecZstd", "0.8.6"),
    ("CSV", "0.10.15"),
    ("DataFrames", "1.7.0"),
    ("HTTP", "1.10.10"),
    ("JLD2", "0.5.8"),
    ("JSON", "0.21.4"),
    ("Memoize", "0.4.4"),
    ("Msgpack", "1.2.1"),
    ("NBInclude", "2.4.0"),
    ("NNlib", "0.9.24"),
    ("Optim", "1.10.0"),
    ("Oxygen", "1.5.14"),
    ("PackageCompiler", "2.1.22"),
    ("Setfield", "1.1.1"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
