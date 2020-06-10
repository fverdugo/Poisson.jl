# This script is to be executed from the root of the project
julia --project=.. -O3 --check-bounds=no --color=yes compile.jl
