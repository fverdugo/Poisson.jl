using PackageCompiler

create_sysimage(:Poisson,
  sysimage_path=joinpath(@__DIR__,"..","Poisson.so"),
  precompile_execution_file=joinpath(@__DIR__,"..","test","runtests.jl"))
