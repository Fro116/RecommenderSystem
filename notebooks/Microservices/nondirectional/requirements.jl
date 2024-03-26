import Pkg
Pkg.develop(path="App")
Pkg.add(; name="Oxygen", version="1.5.3")
Pkg.add(; name="PackageCompiler", version="2.1.17")
import PackageCompiler

PackageCompiler.create_sysimage(
    [
        "Oxygen",
        "App",
    ];
    sysimage_path="sysimg.so",
    precompile_execution_file="main.jl",
    cpu_target="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"
)