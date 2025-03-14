import PackageCompiler

PackageCompiler.create_sysimage(;
    sysimage_path="sysimg.so",
    precompile_execution_file="main.jl",
    cpu_target="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)",
)