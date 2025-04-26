import Pkg

dependencies = [
    ("CSV", "0.10.15"),
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.6"),
    ("DataFrames", "1.7.0"),
    ("DataStructures", "0.18.22"),
    ("Glob", "1.3.1"),
    ("H5Zblosc", "0.1.2"),
    ("HDF5", "0.17.2"),
    ("HTTP", "1.10.16"),
    ("IJulia", "1.27.0"),
    ("Images", "0.26.2"),
    ("JLD2", "0.5.13"),
    ("JSON3", "1.14.2"),
    ("JupyterFormatter", "0.1.1"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("NBInclude", "2.4.0"),
    ("NNlib", "0.9.30"),
    ("OpenCV", "4.6.1"),
    ("Optim", "1.12.0"),
    ("Oxygen", "1.7.1"),
    ("ProgressMeter", "1.10.4"),
    ("StatsBase", "0.34.4"),
    ("StringDistances", "0.11.3"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
