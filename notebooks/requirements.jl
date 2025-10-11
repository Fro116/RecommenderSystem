import Pkg

dependencies = [
    ("CSV", "0.10.15"),
    ("CSVFiles", "1.0.2"),
    ("CodecZlib", "0.7.8"),
    ("CodecZstd", "0.8.6"),
    ("DataFrames", "1.8.0"),
    ("DataStructures", "0.18.22"),
    ("Glob", "1.3.1"),
    ("H5Zblosc", "0.1.2"),
    ("HDF5", "0.17.2"),
    ("HTTP", "1.10.19"),
    ("HypothesisTests", "v0.11.5"),
    ("IJulia", "1.30.6"),
    ("Images", "0.26.2"),
    ("JLD2", "0.6.2"),
    ("JSON3", "1.14.3"),
    ("JupyterFormatter", "0.1.1"),
    ("LibPQ", "1.18.0"),
    ("Memoize", "0.4.4"),
    ("MsgPack", "1.2.1"),
    ("NNlib", "0.9.31"),
    ("OpenCV", "4.6.1"),
    ("Optim", "1.13.2"),
    ("Oxygen", "1.7.5"),
    ("ProgressMeter", "1.11.0"),
    ("StatsBase", "0.34.6"),
    ("StringDistances", "0.11.3"),
]

for (d, v) in dependencies
    Pkg.add(name = d, version = v)
end
