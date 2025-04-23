import Pkg

dependencies = [
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.6"),
    ("CSV", "0.10.15"),
    ("DataFrames", "1.7.0"),
    ("DataStructures", "0.18.22"),
    ("Glob", "1.3.1"),
    ("H5Zblosc", "0.1.2"),
    ("HDF5", "0.17.2"),
    ("HTTP", "1.10.14"),
    ("IJulia", "1.26.0"),
    ("Images", "0.26.2"),
    ("JLD2", "0.5.10"),
    ("JSON", "0.21.4"),
    ("JSON3", "1.14.1"),
    ("JupyterFormatter", "0.1.1"),
    ("LibPQ", "1.18.0"),
    ("MLUtils", "0.4.4"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("NBInclude", "2.4.0"),
    ("NNlib", "0.9.26"),
    ("OpenCV", "4.6.1"),
    ("Optim", "1.10.0"),
    ("Oxygen", "1.5.15"),
    ("ProgressMeter", "1.10.2"),
    ("Setfield", "1.1.1"),
    ("StatsBase", "0.34.3"),
    ("StringDistances", "0.11.3"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
